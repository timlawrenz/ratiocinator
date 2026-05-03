import asyncio
import os
import sys
import time
import re
from ratiocinator.infra.remote import RemoteExecutor

async def monitor():
    from dotenv import load_dotenv
    load_dotenv('/home/tim/source/activity/ratiocinator/.env')
    
    remote = RemoteExecutor("ssh6.vast.ai", 34270, "/home/tim/.ssh/id_rsa")
    
    print("=" * 80)
    print("VAST.AI ABLATION MONITOR")
    print("=" * 80)
    print(f"{'Arm':<5} | {'Step':<6} | {'%':<5} | {'Loss':<8} | {'s/it':<7} | {'ETA':<8}")
    print("-" * 65)

    for arm in ['A', 'B', 'C', 'D']:
        # Fetch the last 100 lines and use python to split by \r to get the actual last tqdm update
        res = await remote.run(f"tail -n 100 /workspace/arm{arm}.log 2>/dev/null", timeout=30)
        logs = res.stdout if hasattr(res, 'stdout') else ""
        
        # Split by both newline and carriage return
        lines = []
        for l in logs.split('\n'):
            lines.extend(l.split('\r'))
            
        train_lines = [l for l in lines if 'Training:' in l]
        line = train_lines[-1] if train_lines else ""
        
        if line:
            step_m = re.search(r"(\d+)/5000", line)
            loss_m = re.search(r"loss=([\d.]+)", line)
            sit_m = re.search(r"([\d.]+)s/it", line)
            
            step = int(step_m.group(1)) if step_m else 0
            pct = step / 5000.0 * 100
            loss = loss_m.group(1) if loss_m else "?"
            sit = sit_m.group(1) if sit_m else "?"
            
            eta = "?"
            if sit != "?" and step < 5000:
                remaining_s = (5000 - step) * float(sit)
                eta = f"{remaining_s / 3600:.1f}h"
                
            print(f"{arm:<5} | {step:<6} | {pct:4.1f}% | {loss:<8} | {sit:<7} | {eta:<8}")
        else:
            print(f"{arm:<5} | Initializing or not running")

if __name__ == '__main__':
    asyncio.run(monitor())
