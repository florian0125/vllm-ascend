"""Placement parity and edge cases, runnable with plain CPU torch."""
import importlib.util
import ctypes
from types import SimpleNamespace
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[3] / "vllm_ascend/expert_offload"


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


frequency = load_module("frequency", "frequency_transfer_policy.py")
with patch.dict(sys.modules, {"vllm_ascend.expert_offload.frequency_transfer_policy": frequency}):
    device = load_module("device_transfer", "device_transfer.py")


class DevicePolicyTests(unittest.TestCase):
    def test_frequency_tie_uses_route_order(self):
        policy = device.DeviceExpertPolicy(4, 2, "cpu", "frequency")
        mapping = torch.tensor([0, 1, -1, -1], dtype=torch.int32)
        eid, slot, active = policy.select(torch.tensor([[3, 2, 3, 2]]), mapping)
        self.assertTrue(active)
        self.assertEqual((int(eid), int(slot)), (3, 0))

    def test_no_active_slot_is_evicted(self):
        policy = device.DeviceExpertPolicy(4, 2, "cpu", "first_miss")
        mapping = torch.tensor([0, 1, -1, -1], dtype=torch.int32)
        original = mapping.clone()
        _, _, active = policy.select(torch.tensor([[0, 1, 2]]), mapping)
        self.assertFalse(active)
        self.assertTrue(torch.equal(mapping, original))

    def test_invalid_routes_and_hits_do_not_upload(self):
        policy = device.DeviceExpertPolicy(3, 2, "cpu", "first_miss")
        mapping = torch.tensor([0, 1, -1], dtype=torch.int32)
        _, _, active = policy.select(torch.tensor([[-1, 0, 8]]), mapping)
        self.assertFalse(active)
        self.assertEqual(policy.counts.tolist(), [1, 0, 0])

    def test_frequency_matches_cpu_over_replacements(self):
        policy = device.DeviceExpertPolicy(16, 4, "cpu", "frequency")
        reference = frequency.FrequencyTransferPolicy()
        mapping = torch.tensor([0, 1, 2, 3] + [-1] * 12, dtype=torch.int32)
        expected_map = mapping.clone()
        generator = torch.Generator().manual_seed(41)
        for step in range(200):
            ids = torch.randint(-1, 16, (2, 4), generator=generator)
            expected = reference.select(ids, expected_map, 4)
            eid, slot, active = policy.select(ids, mapping)
            actual = (int(eid), int(slot)) if active else None
            self.assertEqual(actual, expected, f"step {step}")
            if expected:
                eid, slot = expected
                expected_map[expected_map == slot] = -1
                expected_map[eid] = slot
            self.assertTrue(torch.equal(mapping, expected_map))
            self.assertTrue(torch.allclose(policy.scores, torch.tensor(reference.scores)))


if __name__ == "__main__":
    unittest.main()


def test_host_mapping_aligns_registration_and_preserves_tensor_offset():
    calls = []

    def register(pointer, length, flag, output):
        calls.append((pointer, length, flag))
        ctypes.cast(output, ctypes.POINTER(ctypes.c_void_p)).contents.value = 0x9000
        return 0

    copier = object.__new__(device.DeviceSparseCopy)
    copier.acl = SimpleNamespace(aclrtHostRegister=register,
                                aclrtHostUnregister=lambda pointer: calls.append(pointer) or 0)
    copier.owners, copier.registered = [], []
    tensor = SimpleNamespace(data_ptr=lambda: 0x1234, numel=lambda: 5000, element_size=lambda: 1)
    with patch.object(device.os, 'sysconf', return_value=4096):
        assert copier.map_host(tensor) == 0x9234
    assert calls == [(0x1000, 8192, 0)]
    assert copier.owners == [tensor]
    copier.close()
    assert calls[-1] == 0x1000
    assert not copier.owners and not copier.registered


def test_host_mapping_coalesces_shared_boundary_pages():
    calls = []

    def register(pointer, length, flag, output):
        calls.append((pointer, length, flag))
        ctypes.cast(output, ctypes.POINTER(ctypes.c_void_p)).contents.value = 0x9000
        return 0

    copier = object.__new__(device.DeviceSparseCopy)
    copier.acl = SimpleNamespace(aclrtHostRegister=register,
                                aclrtHostUnregister=lambda pointer: calls.append(pointer) or 0)
    copier.owners, copier.registered = [], []
    tensors = [SimpleNamespace(data_ptr=lambda: 0x1234, numel=lambda: 5000, element_size=lambda: 1),
               SimpleNamespace(data_ptr=lambda: 0x2900, numel=lambda: 4096, element_size=lambda: 1)]
    with patch.object(device.os, 'sysconf', return_value=4096):
        assert copier.map_hosts(tensors) == [0x9234, 0xA900]
    assert calls == [(0x1000, 12288, 0)]
    copier.close()
    assert calls[-1] == 0x1000


def test_host_mapping_preserves_adjacent_allocation_boundaries():
    calls = []

    def register(pointer, length, flag, output):
        calls.append((pointer, length))
        ctypes.cast(output, ctypes.POINTER(ctypes.c_void_p)).contents.value = 0x9000 + len(calls) * 0x10000
        return 0

    copier = object.__new__(device.DeviceSparseCopy)
    copier.acl = SimpleNamespace(aclrtHostRegister=register, aclrtHostUnregister=lambda pointer: 0)
    copier.owners, copier.registered = [], []
    tensors = [SimpleNamespace(data_ptr=lambda: 0x1000, numel=lambda: 4096, element_size=lambda: 1),
               SimpleNamespace(data_ptr=lambda: 0x2000, numel=lambda: 4096, element_size=lambda: 1)]
    with patch.object(device.os, 'sysconf', return_value=4096):
        assert copier.map_hosts(tensors) == [0x19000, 0x29000]
    assert calls == [(0x1000, 4096), (0x2000, 4096)]
    copier.close()


def test_copy_library_recipe_version_supports_legacy_and_tiled():
    for exported, expected in ((None, 1), (2, 2)):
        library = SimpleNamespace(AccOffloadSparseCopy=lambda *args: None,
                                  AccOffloadNativeCopy=lambda *args: None)
        if exported is not None:
            library.AccOffloadNativeCopyPlanVersion = lambda: exported
        acl = SimpleNamespace(aclrtHostRegister=lambda *args: 0,
                              aclrtHostUnregister=lambda *args: 0)
        with patch.object(device.Path, 'is_file', return_value=True), \
             patch.object(device.ctypes, 'CDLL', side_effect=[library, acl]):
            copier = device.DeviceSparseCopy('/unused/library.so')
        assert copier.native_plan_version == expected


def test_copy_library_rejects_unknown_recipe_version():
    library = SimpleNamespace(AccOffloadSparseCopy=lambda *args: None,
                              AccOffloadNativeCopy=lambda *args: None,
                              AccOffloadNativeCopyPlanVersion=lambda: 99)
    with patch.object(device.Path, 'is_file', return_value=True), \
         patch.object(device.ctypes, 'CDLL', return_value=library), \
         unittest.TestCase().assertRaisesRegex(ValueError, 'recipe version'):
        device.DeviceSparseCopy('/unused/library.so')
