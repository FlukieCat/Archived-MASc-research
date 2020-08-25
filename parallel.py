#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Provides runFunctionsInParallel(), for  managing a list of jobs (python functions) that are to be run in parallel.

 - monitor progress of a bunch of function calls, running in parallel
 - capture the output of each function call. This is a problem because Queues normally break if they get too full. Thus we regularly empty them.
 - Close the queues as functions finish. This is key because otherwise the OS shuts us down for using too many open files.

Job results (returned values) should be pickleable.
"""
from datetime import datetime
import multiprocessing as mp
import numpy as np
# from os import nice
import gc # Effort to close files (queues) when done... 
from time import sleep
import time
from math import sqrt

__author__ = "Christopher Barrington-Leigh"
class pWrapper(): # Enclosing in a class originally motivated to make Garbage Collection work better
    def __init__(self,thefunc,theArgs=None,thekwargs=None,delay=None,name=None):
        self.callfunc = thefunc
        self.callargs = theArgs if theArgs is not None else []
        self.callkwargs = thekwargs if thekwargs is not None  else {}
        #self.calldelay=delay  # Or should this be dealt with elsewhere?
        self.name=name  # Or should this be dealt with elsewhere?
        self.funcName ='(built-in function)' if not hasattr(self.callfunc,'__name__') else self.callfunc.__name__
        self.gotQueue=None # collected output from the process
        self.started=False
        self.running=False
        self.finished=False
        self.exitcode='dns' #"Did not start"
        self.is_alive='dns' # For internal use only. Present "running"
        self.queue= 0  # Uninitiated queue is 0. Complete/closed queue will be None

    def get_func(self):
        return self.callfunc
    @staticmethod
    def add_to_queue(thefunc,que,theArgs=None,thekwargs=None,delay=None):
        """ This actually calls one job (function), with its arguments.
        To keep this method static, we avoid reference to class features """
        if delay:
            from time import sleep
            sleep(delay)
        funcName='(built-in function)' if not hasattr(thefunc,'func_name') else thefunc.func_name
        theArgs=theArgs if theArgs is not None else []
        kwargs=thekwargs if thekwargs is not None  else {}
        returnVal=que.put(thefunc(*theArgs,**kwargs))
        # print('INPARALLEL: Finished %s in parallel! '%funcName)
        return(returnVal) #this should be 0.
    def start(self):
        """ Create queue, add to it by calling add_to_queue"""
        assert self.started==False
        self.queue= mp.Queue()
        self.thejob = mp.Process(target=self.add_to_queue, args=[self.callfunc, self.queue,self.callargs,self.callkwargs],)
       
        self.thejob.start()
        # print('INPARALLEL: Launching %s in parallel %s'%(self.funcName,self.name))
        self.started=True
        self.running=True
    def status(self):
        """ Get status of job, and empty the queue if there is something in it """
        if self.started is False:
            return('dns')
        if self.finished:
            return({0:'0',1:'failed'}.get(self.exitcode, self.exitcode))
        assert self.running
        self.is_alive =self.thejob.is_alive()
        cleanup=  self.is_alive not in ['dns',1]
        assert self.running
        # Update/empty queue
        if not self.queue.empty():
            if self.gotQueue is None:
                self.gotQueue=self.queue.get()
            else:
                self.gotQueue+=self.queue.get()
        # Terminate the job, close the queue, try to initiate Garbage Collecting in order to avoid "Too many open files"
        if cleanup: # The following is intended to get arround OSError: [Errno 24] Too many open files.  But it does not. What more can I do to garbage clean the completed queues and jobs?
            self.cleanup()
        return('running')
    def cleanup(self):
        """ Attempts to free up memory and processes after one is finished, so OS doesn't run into problems to do with too many processes.   """
        self.exitcode = self.thejob.exitcode
        self.thejob.join()
        self.thejob.terminate()
        self.queue.close()
        self.thejob=None
        self.queue=None
        self.finished=True
        self.running=False
    def queuestatus(self):
        """ Check whether queue has overflowed """
        if self.queue in [0]:
            return('dns') # Did not start yet
        if self.queue is None:
            return('') # Closed
        return('empty'*self.queue.empty()  + 'full'*self.queue.full() )

def runFunctionsInParallel(*args, **kwargs):
    """ This is the main/only interface to class cRunFunctionsInParallel. See its documentation for arguments.
    """
    if not args[0]:
        return([])
    
    return cRunFunctionsInParallel(*args, **kwargs).launch_jobs()

###########################################################################################
###
class cRunFunctionsInParallel():
    ###
    #######################################################################################
    """Run any list of functions, each with any arguments and keyword-arguments, in parallel.
The functions/jobs should return (if anything) pickleable results. In order to avoid processes getting stuck due to the output queues overflowing, the queues are regularly collected and emptied.

You can now pass os.system or etc to this as the function, in order to parallelize at the OS level, with no need for a wrapper: I made use of hasattr(builtinfunction,'func_name') to check for a name.

Parameters
----------

listOf_FuncAndArgLists : a list of lists 
    List of up-to-three-element-lists, like [function, args, kwargs],
    specifying the set of functions to be launched in parallel.  If an
    element is just a function, rather than a list, then it is assumed
    to have no arguments or keyword arguments. Thus, possible formats
    for elements of the outer list are:
      function
      [function, list]
      [function, list, dict]

kwargs: dict
    One can also supply the kwargs once, for all jobs (or for those
    without their own non-empty kwargs specified in the list)

names: an optional list of names to identify the processes.
    If omitted, the function name is used, so if all the functions are
    the same (ie merely with different arguments), then they would be
    named indistinguishably

offsetsSeconds: int or list of ints
    delay some functions' start times

allowJobFailure: True/False  [This parameter used to be called expectNonzeroExit)
    Normal behaviour is to not proceed if any function exits with a
    failed exit code. This can be used to override this behaviour.

parallel: True/False
    Whenever the list of functions is longer than one, functions will
    be run in parallel unless this parameter is passed as False

maxAtOnce: int
    If nonzero, this limits how many jobs will be allowed to run at
    once.  By default, this is set according to how many processors
    the hardware has available.

showFinished : int

    Specifies the maximum number of successfully finished jobs to show
    in the text interface (before the last report, which should always
    show them all).

Returns
-------

Returns a tuple of (return codes, return values), each a list in order of the jobs provided.

Issues
-------

Only tested on POSIX OSes.

Examples
--------

See the testParallel() method in this module

    """

    def __init__(self,listOf_FuncAndArgLists, kwargs=None, names=None, parallel=None, offsetsSeconds=None, allowJobFailure=False, expectNonzeroExit=False, maxAtOnce=None, showFinished=20, monitor_progress=True):

        self.parallel= mp.cpu_count() >2  if parallel is None or parallel is True  else  parallel # Use parallel only when we have many processing cores (well, here, more than 8)

    
        if not listOf_FuncAndArgLists:
            return # list of functions to run was empty.

        if offsetsSeconds is None:
            offsetsSeconds=0

        # Jobs may be passed as a function, not a list of [function, args, kwargs]:
        listOf_FuncAndArgLists=[faal if isinstance(faal,list) else [faal,[],{}] for faal in listOf_FuncAndArgLists]
        # Jobs may be passed with kwargs missing:
        listOf_FuncAndArgLists=[faal+[{}] if len(faal)==2 else faal for faal in listOf_FuncAndArgLists]
        # Jobs may be passed with both args and kwargs missing:
        listOf_FuncAndArgLists=[faal+[[],{}] if len(faal)==1 else faal for faal in listOf_FuncAndArgLists]
        # kwargs may be passed once to apply to all functions
        kwargs=kwargs if kwargs else [faal[2] for faal in listOf_FuncAndArgLists]

        if len(listOf_FuncAndArgLists)==1:
            self.parallel=False

        if names is None:
            names=[None for fff in listOf_FuncAndArgLists]
        self.names=[names[iii] if names[iii] is not None else fff[0].func_name for iii,fff in enumerate(listOf_FuncAndArgLists)]
        self.funcNames = ['(built-in function)' if not hasattr(afunc,'func_name') else afunc.func_name for afunc,bb,cc in listOf_FuncAndArgLists]
        
        assert len(self.names)==len(listOf_FuncAndArgLists)

        if maxAtOnce is None:
            self.maxAtOnce=max(1,mp.cpu_count()-2) 
        else:
            self.maxAtOnce=max(min(mp.cpu_count()-2,maxAtOnce),1)
            
        # For initial set of launched processes, stagger them with a spacing of the offsetSeconds.
        self.delays= list((  (np.arange(len(listOf_FuncAndArgLists))-1) * ( np.arange(len(listOf_FuncAndArgLists))< self.maxAtOnce  ) + 1 )* offsetsSeconds)
        self.offsetsSeconds = offsetsSeconds
        self.showFinished = showFinished
        if expectNonzeroExit:
            assert not allowJobFailure
            print('parallel.py: expectNonzeroExit is deprecated. Use (identical) allowJobFailure instead')
            allowJobFailure =expectNonzeroExit 
        self.allowJobFailure=  allowJobFailure
        
        # nice(10) # Add 10 to the niceness of this process (POSIX only)

        self.jobs = None
        self.gotQueues=dict()
        self.status=[None for             ii,fff in enumerate(listOf_FuncAndArgLists)]
        self.exitcodes=[None for          ii,fff in enumerate(listOf_FuncAndArgLists)]
        self.queuestatus=[None for        ii,fff in enumerate(listOf_FuncAndArgLists)]

        self.listOf_FuncAndArgLists=listOf_FuncAndArgLists
        self.monitor_progress =  monitor_progress # If False, only report at the end.
    def run(self): # Just a shortcut
        return self.launch_jobs()

    def launch_jobs(self):

        if self.parallel is False:
            print('++++++++++++++++++++++  DOING FUNCTIONS SEQUENTIALLY ---------------- (parallel=False in runFunctionsInParallel)')
            returnVals=[fffargs[0](*(fffargs[1]),**(fffargs[2]))  for iffargs,fffargs in enumerate(self.listOf_FuncAndArgLists)]
            # In non-parallel a job failure should abort before returning. So we don't care about return values.
            return(returnVals)

        
        """ Use pWrapper class to set up and launch jobs and their queues. Issue reports at decreasing frequency. """
        self.jobs = [pWrapper(funcArgs[0],funcArgs[1],funcArgs[2],self.delays[iii],self.names[iii]) for iii,funcArgs in enumerate(self.listOf_FuncAndArgLists)]
        # [Attempting to avoid running into system limits] Let's never create a loop variable which takes on the value of an element of the above list. Always instead dereference the list using an index.  So no local variables take on the value of a job. (In addition, the job class is supposed to clean itself up when a job is done running).

        istart= self.maxAtOnce if self.maxAtOnce<len(self.jobs) else len(self.jobs)
        for ijob in range(istart):
            self.jobs[ijob].start() # Launch them all

        timeElapsed=0

        self.updateStatus()
        # if 0: self.reportStatus(status, exitcodes,names,istart,showFinished) # This is not necessary; we can leave it to the first loop, below, to report. But for debug, this shows the initial batch.

        """ Now, wait for all the jobs to finish.  Allow for everything to finish quickly, at the beginning. 
        """
        lastreport=''
        while any([self.status[ijj]=='running' for  ijj in range(len(self.jobs))]) or istart<len(self.jobs):
            sleepTime=5*(timeElapsed>2)
            if timeElapsed>0:
                time.sleep(1+sleepTime) # Wait a while before next update. Slow down updates for really long runs.
            timeElapsed+=sleepTime
            # Add any extra jobs needed to reach the maximum allowed:
            newjobs=0
            while istart<len(self.jobs) and sum([self.status[ijj] in ['running'] for ijj in range(len(self.jobs))]) < self.maxAtOnce:
                self.jobs[istart].start()
                newjobs+=1
                self.updateStatus()
                if newjobs>=self.maxAtOnce:
                    # lastreport= self.reportStatus(istart, previousReportString=lastreport)
                    newjobs=0
                istart+=1
                timeElapse=.01

            self.updateStatus()
            # lastreport= self.reportStatus(istart, previousReportString=lastreport) #self.reportStatus(status, exitcodes,names,istart,showFinished,  previousReportString=lastreport)

        # All jobs are finished. Give final report of exit statuses
        self.updateStatus()
        # self.reportStatus( np.inf)
        if any(self.exitcodes):
            print('INPARALLEL: Parallel processing batch set did not ALL succeed successfully ('+' '.join(self.names)+')')
            assert self.allowJobFailure  # one of the functions you called failed.
            print('          Tolerating job failure since allowJobFailure == True')
            return(False)
        # else:
        #     print('INPARALLEL: Apparent success of all functions ('+' '.join(self.names)+')')
        return(self.exitcodes,[self.gotQueues[ii] for ii in range(len(self.jobs))])


    def updateStatus(self):
        for ii in range(len(self.jobs)):
            if self.status[ii] not in ['failed','success','0',0,1,'1']: 
                self.status[ii]=self.jobs[ii].status()
                self.exitcodes[ii]=self.jobs[ii].exitcode
                self.queuestatus[ii]=self.jobs[ii].queuestatus()
            if self.status[ii] not in ['dns','running',None] and ii not in self.gotQueues:
                    self.gotQueues[ii]=self.jobs[ii].gotQueue
                    #jobs[ii].destroy()
                    self.jobs[ii]=None
                    gc.collect()

    def reportStatus(self, showmax, showsuccessful=np.inf, previousReportString=''):
        """
        """
        if not self.monitor_progress: return('')
        outs=''
        ishowable=range(min(len(self.status), showmax))
        istarted=[ii for ii in range(len(self.status)) if  self.status[ii] not in ['dns']]
        isuccess=[ii for ii in ishowable if self.status[ii] in ['success',0,'0']]
        irunning=[ii for ii in range(len(self.status)) if  self.status[ii] in ['running']]
        earliestSuccess= -1 if len(isuccess)<showsuccessful else isuccess[::-1][showsuccessful-1]
        if 0:
            print(showmax,showsuccessful,earliestSuccess)
            print(len(isuccess)-showsuccessful)
        max_name_length = max([len(name) for name in self.names])
        max_funcname_length = max([len(name) for name in self.funcNames])
        tableFormatString='%'+str(max_name_length)+'s:\t%10s\t%10s\t%s()'
        outs+= '-'*(max_name_length+12+max_funcname_length)+'\n'+ tableFormatString%('Job','Status','Queue','Func',)+ '\n'+'-'*(max_name_length+12+max_funcname_length)+'\n'
        # Check that we aren't going to show more *successfully finished* jobs than we're allowed: Find index of nth-last successful one. That is, if the limit binds, we should show the latest N=showsuccessful ones only.
        outs+=  '\n'.join([tableFormatString%(self.names[ii],self.status[ii], self.queuestatus[ii], self.funcNames[ii]) for ii in ishowable if self.status[ii] not in ['success',0,'0'] or ii>=earliestSuccess  ]) + '\n'

                          #'' if self.jobs[ii] is None else '(built-in function)' if not hasattr(self.jobs[ii].get_func(),'func_name') else self.jobs[ii].get_func().func_name)
        if len(isuccess)>showsuccessful: # We don't hide failed jobs, but we do sometimes skip older successful jobs
            outs+=   '%d job%s running. %d other jobs finished successfully.\n'%(len(irunning), 's'*(len(irunning)!=1), len(isuccess)-showsuccessful)
        else:
            outs+=   '%d job%s running.\n' % (len(irunning),'s'*(len(irunning)!=1))
        if len(self.status)>len(istarted):
            outs+=   '%d more jobs waiting for their turn to start...\n'%(len(self.status)-len(istarted)) ##len(sjobs)-len(djobs))
        #print('%d open queues...'%len(queues))
        outs+= '-'*(max_name_length+12+max_funcname_length)+'\n'
        #return([exitcode(job) for ii,job in enumerate(sjobs)])
        if outs != previousReportString:
            print('\n'+datetime.now().strftime("%Y-%m-%d %H:%M:%S")            )
            print(outs+'\n')
        return(outs)

    def emptyQueues(self):#jobs,queues,gotQueues):
        for ii,job in enumerate(self.jobs):
            if ii not in self.queues or not isinstance(self.queues[ii],mp.queues.Queue):
                continue
            cleanup= self.exitcodes(job)==0
            
            if not self.queues[ii].empty():
                if ii in gotQueues:
                    self.gotQueues[ii]+= self.queues[ii].get()
                else:
                    self.gotQueues[ii]= self.queues[ii].get()
            if cleanup: # The following is intended to get arround OSError: [Errno 24] Too many open files.  But it does not. What more can I do to garbage clean the completed queues and jobs?
                job.join()
                job.terminate()
                self.queues[ii].close()
                """
        print('Joined job %d'%ii)
        job.terminate()
        print('Terminated job %d'%ii)
        queues[ii].close()
                """
                job=None
                #del job
                self.queues[ii]=None
                #del queues[ii] # This seems key. Before, when I kept queues in a list, deleting the item wasn't good enough.
                #print('                       Cleaning up/closing queue for job %d'%ii)
                


def breaktest(): # The following demonstrates how to clean up jobs and queues (the queues was key) to avoid the OSError of too many files open. But why does this not work, above? Because there's still a pointer in the list of queues? No, 
    def dummy(inv,que):
        que.put(inv)
        return(0)
    from multiprocessing import Process, Queue
    nTest=1800
    queues=[None for ii in range(nTest)]
    jobs=[None for ii in range(nTest)]#[Process(target=dummy, args=[ii,queues[ii]]) for ii in range(nTest)]
    #for ii,job in enumerate(jobs):
    for ii in range(nTest):#,job in enumerate(jobs):
        queues[ii]=Queue()
        job=Process(target=dummy, args=[ii,queues[ii]])
        job.start()
        print('Started job %d'%ii)
        job.join()
        print('Joined job %d'%ii)
        job.terminate()
        print('Terminated job %d'%ii)
        queues[ii].close()
        queues[ii]=None #  This line does it!

def return_values():
    """ DataFrames,for examples, don't have a boolean value.
What can we do to discern failures from return values?
    """
        
def test_allowJobFailure(): # In parallel==True case
    try:
        runFunctionsInParallel([[lambda: 1/0], [lambda: 1/0]])
        raise(Exception('Test failed'))
    except AssertionError:
        print(' ----> runFunctionsInParallel correctly objected to job failure.')
    try:
        runFunctionsInParallel([[lambda: 1/0], [lambda: 1/0]], allowJobFailure=True)
        print(' ----> runFunctionsInParallel correctly tolerated job failure.')
    except AssertionError:
        raise(Exception('Test failed'))
    return

def testParallel():
    import numpy as np

    # Demo longer jobs, since other demos' jobs finish too quickly on some platforms
    # def doodle4():
    #     ll=np.random.randint(7)+3
    #     i=0
    #     while i<10**ll:
    #         i=i+1
    #     return(i)
    # nTest=20
    # runFunctionsInParallel([doodle4 for ii in range(nTest)],names=[str(ii) for ii in range(nTest)], offsetsSeconds=None, maxAtOnce=10, showFinished=5)

    # Test use of kwargs
    def doodle1(jj, a=None, b=None):
        i=0 + len(str(a))+len(str(b))
        while i<1e2:
            i=i+1
        return(jj)
    nTest=10
    runFunctionsInParallel([[doodle1,[ii],{'a':5,'b':10}] for ii in range(nTest)],names=[str(ii) for ii in range(nTest)], offsetsSeconds=0.2, maxAtOnce=40, parallel=True, allowJobFailure=True)

    # Demo simpler use, function takes no arguments
    def doodle2():
        i=0
        while i<1e9:
            i=i+1
        return(i)
    nTest=100
    runFunctionsInParallel([doodle2 for ii in range(nTest)],names=[str(ii) for ii in range(nTest)], offsetsSeconds=0.2, maxAtOnce=10, showFinished=5)

    
    # Test use of large number of jobs, enough to make some systems get upset without our countermeasures
    def doodle3(jj, a=None, b=None):
        i=0
        while 0:#i<1e2:
            i=i+1
        return(jj)
    nTest=2700
    runFunctionsInParallel([[doodle3,[ii],{'a':5,'b':10}] for ii in range(nTest)],names=[str(ii) for ii in range(nTest)], offsetsSeconds=0.2, maxAtOnce=40, parallel=True, allowJobFailure=True)



################################################################################################
if __name__ == '__main__':
################################################################################################
    testParallel()
