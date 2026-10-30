import json,os,time
from pathlib import Path
import torch
_stream=None
def record_clock_calibration(tag,directory):
    global _stream
    if _stream is None:_stream=torch.npu.Stream()
    torch.npu.synchronize()
    event=torch.npu.Event(enable_timing=True)
    event.record(_stream);event.synchronize()
    samples=[]
    for index in range(16):
        before=time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
        event.record(_stream);event.synchronize()
        after=time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
        samples.append(dict(index=index,before_raw_ns=before,after_raw_ns=after))
    with (Path(directory)/f'cpu_clock_calibration_{os.getpid()}.jsonl').open('a') as f:
        f.write(json.dumps(dict(tag=tag,pid=os.getpid(),samples=samples))+'\n')
