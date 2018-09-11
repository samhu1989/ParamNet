import os;
import sys;
import subprocess;
import psutil;
import datetime;
import gc;

def safe_guard():
    try:
        pro = print_info();
        check_open_max(pro);
        check_mem_max(pro);
    except (psutil.ZombieProcess, psutil.AccessDenied, psutil.NoSuchProcess):
        print >>sys.stderr,'Exception in safe guard';
        
def print_info():
    pro = psutil.Process(os.getpid());
    p_name = pro.name();
    p_cpud = pro.cpu_percent(interval=5)
    p_mem = pro.memory_percent();
    print>>sys.stderr,datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]:"),"name:",p_name,",cpu:",p_cpud,",mem:",p_mem,",files:",len(pro.open_files());
    return pro;
    
def check_open_max(pro=None,max=1000):
    if pro is None:
        pro = psutil.Process(os.getpid());
    open_num = len(pro.open_files());
    if open_num >= max:
        print >>sys.stderr,"Exit:open %d files, which is too much(>=%d)"%(open_num,max);
        sys.exit();
        
def check_mem_max(pro=None,max=30.0):
    if pro is None:
        pro = psutil.Process(os.getpid());
    if pro.memory_percent() > max//2:
        gc.collect();
    if pro.memory_percent() > max:
        print >>sys.stderr,"%f mem used, which is too much(>=%f)"%(pro.memory_percent(),max);
        sys.exit();
        

        