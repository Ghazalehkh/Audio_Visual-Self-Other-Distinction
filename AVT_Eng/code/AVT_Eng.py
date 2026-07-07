#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2026.1.0),
    on July 07, 2026, at 13:10
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (
    NOT_STARTED, STARTED, PLAYING, PAUSED, STOPPED, STOPPING, FINISHED, PRESSED, 
    RELEASED, FOREVER, priority
)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard

# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2026.1.0'
expName = 'mYousefi'  # from the Builder filename that created this script
expVersion = ''
# a list of functions to run when the experiment ends (starts off blank)
runAtExit = []
# information about this experiment
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '001',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'expVersion|hid': expVersion,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = True
_winSize = (1024, 768)
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']
    # replace default participant ID
    if prefs.piloting['replaceParticipantID']:
        expInfo['participant'] = 'pilot'

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version=expVersion,
        extraInfo=expInfo, runtimeInfo=None,
        originPath='G:\\Ghazaleh Khosravi\\Task Design Psychopy\\AVT_Eng_github\\code\\mYousefi.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    # store pilot mode in data file
    thisExp.addData('piloting', PILOTING, priority=priority.LOW)
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # set how much information should be printed to the console / app
    if PILOTING:
        logging.console.setLevel(
            prefs.piloting['pilotConsoleLoggingLevel']
        )
    else:
        logging.console.setLevel('warning')
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log')
    if PILOTING:
        logFile.setLevel(
            prefs.piloting['pilotLoggingLevel']
        )
    else:
        logFile.setLevel(
            logging.getLevel('exp')
        )
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if PILOTING:
        logging.debug('Fullscreen settings ignored as running in pilot mode.')
    
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowGUI=False, allowStencil=False,
            monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height',
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    if expInfo is not None:
        # get/measure frame rate if not already in expInfo
        if win._monitorFrameRate is None:
            win._monitorFrameRate = win.getActualFrameRate(infoMsg='Attempting to measure frame rate of screen, please wait...')
        expInfo['frameRate'] = win._monitorFrameRate
    win.hideMessage()
    if PILOTING:
        # show a visual indicator if we're in piloting mode
        if prefs.piloting['showPilotingIndicator']:
            win.showPilotingIndicator()
        # always show the mouse in piloting mode
        if prefs.piloting['forceMouseVisible']:
            win.mouseVisible = True
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    
    # Setup iohub keyboard
    ioConfig['Keyboard'] = dict(use_keymap='psychopy')
    
    # Setup iohub experiment
    ioConfig['Experiment'] = dict(filename=thisExp.dataFileName)
    
    # Start ioHub server
    ioServer = io.launchHubServer(window=win, **ioConfig)
    
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='iohub'
        )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], currentRoutine=None):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    currentRoutine : psychopy.data.Routine
        Current Routine we are in at time of pausing, if any. This object tells PsychoPy what Components to pause/play/dispatch.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # start a timer to figure out how long we're paused for
    pauseTimer = core.Clock()
    # pause any playback components
    if currentRoutine is not None:
        for comp in currentRoutine.getPlaybackComponents():
            comp.pause()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='ioHub',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # dispatch messages on response components
        if currentRoutine is not None:
            for comp in currentRoutine.getDispatchComponents():
                comp.device.dispatchMessages()
        # sleep 1ms so other threads can execute
        clock.time.sleep(0.001)
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    if currentRoutine is not None:
        for comp in currentRoutine.getPlaybackComponents():
            comp.play()
    # reset any timers
    for timer in timers:
        timer.addTime(-pauseTimer.getTime())


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # update experiment info
    expInfo['date'] = data.getDateStr()
    expInfo['expName'] = expName
    expInfo['expVersion'] = expVersion
    expInfo['psychopyVersion'] = psychopyVersion
    # make sure window is set to foreground to prevent losing focus
    win.winHandle.activate()
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ioHub'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "PractPhase" ---
    back8 = visual.Rect(
        win=win, name='back8',units='norm', 
        width=(2,2)[0], height=(2,2)[1],
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='black', fillColor='black',
        opacity=1.0, depth=0.0, interpolate=True)
    practText = visual.TextStim(win=win, name='practText',
        text="Press 'space' to start the practice trial...",
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=1.0, 
        languageStyle='LTR',
        depth=-1.0);
    Resp4Space = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "taskT1" ---
    back3 = visual.Rect(
        win=win, name='back3',units='norm', 
        width=(2,2)[0], height=(2,2)[1],
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='black', fillColor='black',
        opacity=1.0, depth=0.0, interpolate=True)
    OtherText = visual.TextStim(win=win, name='OtherText',
        text='',
        font='Open Sans',
        units='norm', pos=(0,0.75), draggable=False, height=0.09, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=1.0, 
        languageStyle='LTR',
        depth=-1.0);
    Pic = visual.ImageStim(
        win=win,
        name='Pic', units='norm', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0,0), draggable=False, size=(1,1),
        color=[1,1,1], colorSpace='rgb', opacity=1.0,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    # set audio backend
    sound.Sound.backend = 'ptb'
    Vc = sound.Sound(
        'A', 
        secs=-1, 
        stereo=True, 
        hamming=True, 
        speaker=None,    name='Vc'
    )
    Vc.setVolume(1.0)
    
    # --- Initialize components for Routine "Response" ---
    back6 = visual.Rect(
        win=win, name='back6',units='norm', 
        width=(2,2)[0], height=(2,2)[1],
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='black', fillColor='black',
        opacity=1.0, depth=0.0, interpolate=True)
    Quest = visual.TextStim(win=win, name='Quest',
        text='',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=1.0, 
        languageStyle='LTR',
        depth=-1.0);
    RatingResp = visual.Slider(win=win, name='RatingResp',
        startValue=None, size=(1.65, 0.1), pos=(0, -0.3), units=win.units,
        labels=[-10 ,0 , +10], ticks=(-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1, 2, 3, 4, 5,6,7,8,9,10), granularity=1.0,
        style='rating', styleTweaks=(), opacity=1.0,
        labelColor='LightGray', markerColor='Red', lineColor='White', colorSpace='rgb',
        font='Open Sans', labelHeight=0.04,
        flip=False, ori=0.0, depth=-2, readOnly=False)
    
    # --- Initialize components for Routine "Fc" ---
    back2 = visual.Rect(
        win=win, name='back2',units='norm', 
        width=(2,2)[0], height=(2,2)[1],
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='black', fillColor='black',
        opacity=1.0, depth=0.0, interpolate=True)
    FixCross = visual.ShapeStim(
        win=win, name='FixCross', vertices='cross',
        size=(0.25, 0.25),
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=1.0, depth=-1.0, interpolate=True)
    
    # --- Initialize components for Routine "instruction" ---
    background = visual.Rect(
        win=win, name='background',units='norm', 
        width=(2,2)[0], height=(2,2)[1],
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='black', fillColor='black',
        opacity=1.0, depth=0.0, interpolate=True)
    PressText = visual.TextStim(win=win, name='PressText',
        text="Press 'space' if you are ready...",
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=1.0, 
        languageStyle='LTR',
        depth=-1.0);
    Resp1Space = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "Fc" ---
    back2 = visual.Rect(
        win=win, name='back2',units='norm', 
        width=(2,2)[0], height=(2,2)[1],
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='black', fillColor='black',
        opacity=1.0, depth=0.0, interpolate=True)
    FixCross = visual.ShapeStim(
        win=win, name='FixCross', vertices='cross',
        size=(0.25, 0.25),
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=1.0, depth=-1.0, interpolate=True)
    
    # --- Initialize components for Routine "taskT1" ---
    back3 = visual.Rect(
        win=win, name='back3',units='norm', 
        width=(2,2)[0], height=(2,2)[1],
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='black', fillColor='black',
        opacity=1.0, depth=0.0, interpolate=True)
    OtherText = visual.TextStim(win=win, name='OtherText',
        text='',
        font='Open Sans',
        units='norm', pos=(0,0.75), draggable=False, height=0.09, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=1.0, 
        languageStyle='LTR',
        depth=-1.0);
    Pic = visual.ImageStim(
        win=win,
        name='Pic', units='norm', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0,0), draggable=False, size=(1,1),
        color=[1,1,1], colorSpace='rgb', opacity=1.0,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    Vc = sound.Sound(
        'A', 
        secs=-1, 
        stereo=True, 
        hamming=True, 
        speaker=None,    name='Vc'
    )
    Vc.setVolume(1.0)
    
    # --- Initialize components for Routine "Response" ---
    back6 = visual.Rect(
        win=win, name='back6',units='norm', 
        width=(2,2)[0], height=(2,2)[1],
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='black', fillColor='black',
        opacity=1.0, depth=0.0, interpolate=True)
    Quest = visual.TextStim(win=win, name='Quest',
        text='',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=1.0, 
        languageStyle='LTR',
        depth=-1.0);
    RatingResp = visual.Slider(win=win, name='RatingResp',
        startValue=None, size=(1.65, 0.1), pos=(0, -0.3), units=win.units,
        labels=[-10 ,0 , +10], ticks=(-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1, 2, 3, 4, 5,6,7,8,9,10), granularity=1.0,
        style='rating', styleTweaks=(), opacity=1.0,
        labelColor='LightGray', markerColor='Red', lineColor='White', colorSpace='rgb',
        font='Open Sans', labelHeight=0.04,
        flip=False, ori=0.0, depth=-2, readOnly=False)
    
    # --- Initialize components for Routine "Fc" ---
    back2 = visual.Rect(
        win=win, name='back2',units='norm', 
        width=(2,2)[0], height=(2,2)[1],
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='black', fillColor='black',
        opacity=1.0, depth=0.0, interpolate=True)
    FixCross = visual.ShapeStim(
        win=win, name='FixCross', vertices='cross',
        size=(0.25, 0.25),
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=1.0, depth=-1.0, interpolate=True)
    
    # --- Initialize components for Routine "BreakRoutine" ---
    back5 = visual.Rect(
        win=win, name='back5',units='norm', 
        width=(2,2)[0], height=(2,2)[1],
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='black', fillColor='black',
        opacity=1.0, depth=0.0, interpolate=True)
    BreakText = visual.TextStim(win=win, name='BreakText',
        text="Have a break...\npress 'space' if you are ready",
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    Resp2Space = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "Fc" ---
    back2 = visual.Rect(
        win=win, name='back2',units='norm', 
        width=(2,2)[0], height=(2,2)[1],
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='black', fillColor='black',
        opacity=1.0, depth=0.0, interpolate=True)
    FixCross = visual.ShapeStim(
        win=win, name='FixCross', vertices='cross',
        size=(0.25, 0.25),
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=1.0, depth=-1.0, interpolate=True)
    
    # --- Initialize components for Routine "taskT1" ---
    back3 = visual.Rect(
        win=win, name='back3',units='norm', 
        width=(2,2)[0], height=(2,2)[1],
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='black', fillColor='black',
        opacity=1.0, depth=0.0, interpolate=True)
    OtherText = visual.TextStim(win=win, name='OtherText',
        text='',
        font='Open Sans',
        units='norm', pos=(0,0.75), draggable=False, height=0.09, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=1.0, 
        languageStyle='LTR',
        depth=-1.0);
    Pic = visual.ImageStim(
        win=win,
        name='Pic', units='norm', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0,0), draggable=False, size=(1,1),
        color=[1,1,1], colorSpace='rgb', opacity=1.0,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    Vc = sound.Sound(
        'A', 
        secs=-1, 
        stereo=True, 
        hamming=True, 
        speaker=None,    name='Vc'
    )
    Vc.setVolume(1.0)
    
    # --- Initialize components for Routine "Response" ---
    back6 = visual.Rect(
        win=win, name='back6',units='norm', 
        width=(2,2)[0], height=(2,2)[1],
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='black', fillColor='black',
        opacity=1.0, depth=0.0, interpolate=True)
    Quest = visual.TextStim(win=win, name='Quest',
        text='',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=1.0, 
        languageStyle='LTR',
        depth=-1.0);
    RatingResp = visual.Slider(win=win, name='RatingResp',
        startValue=None, size=(1.65, 0.1), pos=(0, -0.3), units=win.units,
        labels=[-10 ,0 , +10], ticks=(-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1, 2, 3, 4, 5,6,7,8,9,10), granularity=1.0,
        style='rating', styleTweaks=(), opacity=1.0,
        labelColor='LightGray', markerColor='Red', lineColor='White', colorSpace='rgb',
        font='Open Sans', labelHeight=0.04,
        flip=False, ori=0.0, depth=-2, readOnly=False)
    
    # --- Initialize components for Routine "Fc" ---
    back2 = visual.Rect(
        win=win, name='back2',units='norm', 
        width=(2,2)[0], height=(2,2)[1],
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='black', fillColor='black',
        opacity=1.0, depth=0.0, interpolate=True)
    FixCross = visual.ShapeStim(
        win=win, name='FixCross', vertices='cross',
        size=(0.25, 0.25),
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=1.0, depth=-1.0, interpolate=True)
    
    # --- Initialize components for Routine "Finish" ---
    back7 = visual.Rect(
        win=win, name='back7',units='norm', 
        width=(2,2)[0], height=(2,2)[1],
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='black', fillColor='black',
        opacity=1.0, depth=0.0, interpolate=True)
    TnxParticip = visual.TextStim(win=win, name='TnxParticip',
        text='Thanks for your Participation!',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.09, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=1.0, 
        languageStyle='LTR',
        depth=-1.0);
    Resp3Space = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    if eyetracker is not None:
        eyetracker.enableEventReporting()
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "PractPhase" ---
    # create an object to store info about Routine PractPhase
    PractPhase = data.Routine(
        name='PractPhase',
        components=[back8, practText, Resp4Space],
    )
    PractPhase.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for Resp4Space
    Resp4Space.keys = []
    Resp4Space.rt = []
    _Resp4Space_allKeys = []
    # store start times for PractPhase
    PractPhase.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    PractPhase.tStart = globalClock.getTime(format='float')
    PractPhase.status = STARTED
    thisExp.addData('PractPhase.started', PractPhase.tStart)
    PractPhase.maxDuration = None
    # keep track of which components have finished
    PractPhaseComponents = PractPhase.components
    for thisComponent in PractPhase.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "PractPhase" ---
    thisExp.currentRoutine = PractPhase
    PractPhase.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *back8* updates
        
        # if back8 is starting this frame...
        if back8.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            back8.frameNStart = frameN  # exact frame index
            back8.tStart = t  # local t and not account for scr refresh
            back8.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(back8, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'back8.started')
            # update status
            back8.status = STARTED
            back8.setAutoDraw(True)
        
        # if back8 is active this frame...
        if back8.status == STARTED:
            # update params
            pass
        
        # *practText* updates
        
        # if practText is starting this frame...
        if practText.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            practText.frameNStart = frameN  # exact frame index
            practText.tStart = t  # local t and not account for scr refresh
            practText.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(practText, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'practText.started')
            # update status
            practText.status = STARTED
            practText.setAutoDraw(True)
        
        # if practText is active this frame...
        if practText.status == STARTED:
            # update params
            pass
        
        # *Resp4Space* updates
        waitOnFlip = False
        
        # if Resp4Space is starting this frame...
        if Resp4Space.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            Resp4Space.frameNStart = frameN  # exact frame index
            Resp4Space.tStart = t  # local t and not account for scr refresh
            Resp4Space.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Resp4Space, 'tStartRefresh')  # time at next scr refresh
            # update status
            Resp4Space.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(Resp4Space.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(Resp4Space.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if Resp4Space.status == STARTED and not waitOnFlip:
            theseKeys = Resp4Space.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _Resp4Space_allKeys.extend(theseKeys)
            if len(_Resp4Space_allKeys):
                Resp4Space.keys = _Resp4Space_allKeys[-1].name  # just the last key pressed
                Resp4Space.rt = _Resp4Space_allKeys[-1].rt
                Resp4Space.duration = _Resp4Space_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=PractPhase,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            PractPhase.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if PractPhase.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in PractPhase.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "PractPhase" ---
    for thisComponent in PractPhase.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for PractPhase
    PractPhase.tStop = globalClock.getTime(format='float')
    PractPhase.tStopRefresh = tThisFlipGlobal
    thisExp.addData('PractPhase.stopped', PractPhase.tStop)
    thisExp.nextEntry()
    # the Routine "PractPhase" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    PractTrial = data.TrialHandler2(
        name='PractTrial',
        nReps=1.0, 
        method='fullRandom', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('practice.xlsx'), 
        seed=None, 
        isTrials=False, 
    )
    thisExp.addLoop(PractTrial)  # add the loop to the experiment
    thisPractTrial = PractTrial.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisPractTrial.rgb)
    if thisPractTrial != None:
        for paramName in thisPractTrial:
            globals()[paramName] = thisPractTrial[paramName]
    
    for thisPractTrial in PractTrial:
        PractTrial.status = STARTED
        if hasattr(thisPractTrial, 'status'):
            thisPractTrial.status = STARTED
        currentLoop = PractTrial
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        # abbreviate parameter names if possible (e.g. rgb = thisPractTrial.rgb)
        if thisPractTrial != None:
            for paramName in thisPractTrial:
                globals()[paramName] = thisPractTrial[paramName]
        
        # --- Prepare to start Routine "taskT1" ---
        # create an object to store info about Routine taskT1
        taskT1 = data.Routine(
            name='taskT1',
            components=[back3, OtherText, Pic, Vc],
        )
        taskT1.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        OtherText.setText('The Other Person is listening to...')
        Pic.setImage(image)
        Vc.setSound(vc, secs=duration, hamming=True)
        Vc.setVolume(1.0, log=False)
        Vc.seek(0)
        # store start times for taskT1
        taskT1.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        taskT1.tStart = globalClock.getTime(format='float')
        taskT1.status = STARTED
        thisExp.addData('taskT1.started', taskT1.tStart)
        taskT1.maxDuration = None
        # keep track of which components have finished
        taskT1Components = taskT1.components
        for thisComponent in taskT1.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "taskT1" ---
        thisExp.currentRoutine = taskT1
        taskT1.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisPractTrial, 'status') and thisPractTrial.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *back3* updates
            
            # if back3 is starting this frame...
            if back3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                back3.frameNStart = frameN  # exact frame index
                back3.tStart = t  # local t and not account for scr refresh
                back3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(back3, 'tStartRefresh')  # time at next scr refresh
                # update status
                back3.status = STARTED
                back3.setAutoDraw(True)
            
            # if back3 is active this frame...
            if back3.status == STARTED:
                # update params
                pass
            
            # if back3 is stopping this frame...
            if back3.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > back3.tStartRefresh + duration-frameTolerance:
                    # keep track of stop time/frame for later
                    back3.tStop = t  # not accounting for scr refresh
                    back3.tStopRefresh = tThisFlipGlobal  # on global time
                    back3.frameNStop = frameN  # exact frame index
                    # update status
                    back3.status = FINISHED
                    back3.setAutoDraw(False)
            
            # *OtherText* updates
            
            # if OtherText is starting this frame...
            if OtherText.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                OtherText.frameNStart = frameN  # exact frame index
                OtherText.tStart = t  # local t and not account for scr refresh
                OtherText.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(OtherText, 'tStartRefresh')  # time at next scr refresh
                # update status
                OtherText.status = STARTED
                OtherText.setAutoDraw(True)
            
            # if OtherText is active this frame...
            if OtherText.status == STARTED:
                # update params
                pass
            
            # if OtherText is stopping this frame...
            if OtherText.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > OtherText.tStartRefresh + duration-frameTolerance:
                    # keep track of stop time/frame for later
                    OtherText.tStop = t  # not accounting for scr refresh
                    OtherText.tStopRefresh = tThisFlipGlobal  # on global time
                    OtherText.frameNStop = frameN  # exact frame index
                    # update status
                    OtherText.status = FINISHED
                    OtherText.setAutoDraw(False)
            
            # *Pic* updates
            
            # if Pic is starting this frame...
            if Pic.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                Pic.frameNStart = frameN  # exact frame index
                Pic.tStart = t  # local t and not account for scr refresh
                Pic.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Pic, 'tStartRefresh')  # time at next scr refresh
                # update status
                Pic.status = STARTED
                Pic.setAutoDraw(True)
            
            # if Pic is active this frame...
            if Pic.status == STARTED:
                # update params
                pass
            
            # if Pic is stopping this frame...
            if Pic.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > Pic.tStartRefresh + duration-frameTolerance:
                    # keep track of stop time/frame for later
                    Pic.tStop = t  # not accounting for scr refresh
                    Pic.tStopRefresh = tThisFlipGlobal  # on global time
                    Pic.frameNStop = frameN  # exact frame index
                    # update status
                    Pic.status = FINISHED
                    Pic.setAutoDraw(False)
            
            # *Vc* updates
            
            # if Vc is starting this frame...
            if Vc.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                Vc.frameNStart = frameN  # exact frame index
                Vc.tStart = t  # local t and not account for scr refresh
                Vc.tStartRefresh = tThisFlipGlobal  # on global time
                # update status
                Vc.status = STARTED
                Vc.play(when=win)  # sync with win flip
            
            # if Vc is stopping this frame...
            if Vc.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > Vc.tStartRefresh + duration-frameTolerance or Vc.isFinished:
                    # keep track of stop time/frame for later
                    Vc.tStop = t  # not accounting for scr refresh
                    Vc.tStopRefresh = tThisFlipGlobal  # on global time
                    Vc.frameNStop = frameN  # exact frame index
                    # update status
                    Vc.status = FINISHED
                    Vc.stop()
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=taskT1,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                taskT1.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if taskT1.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in taskT1.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "taskT1" ---
        for thisComponent in taskT1.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for taskT1
        taskT1.tStop = globalClock.getTime(format='float')
        taskT1.tStopRefresh = tThisFlipGlobal
        thisExp.addData('taskT1.stopped', taskT1.tStop)
        Vc.pause()  # ensure sound has stopped at end of Routine
        # the Routine "taskT1" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "Response" ---
        # create an object to store info about Routine Response
        Response = data.Routine(
            name='Response',
            components=[back6, Quest, RatingResp],
        )
        Response.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        Quest.setText(text)
        RatingResp.reset()
        # store start times for Response
        Response.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        Response.tStart = globalClock.getTime(format='float')
        Response.status = STARTED
        thisExp.addData('Response.started', Response.tStart)
        Response.maxDuration = None
        # keep track of which components have finished
        ResponseComponents = Response.components
        for thisComponent in Response.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "Response" ---
        thisExp.currentRoutine = Response
        Response.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 10.0:
            # if trial has changed, end Routine now
            if hasattr(thisPractTrial, 'status') and thisPractTrial.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *back6* updates
            
            # if back6 is starting this frame...
            if back6.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                back6.frameNStart = frameN  # exact frame index
                back6.tStart = t  # local t and not account for scr refresh
                back6.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(back6, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'back6.started')
                # update status
                back6.status = STARTED
                back6.setAutoDraw(True)
            
            # if back6 is active this frame...
            if back6.status == STARTED:
                # update params
                pass
            
            # if back6 is stopping this frame...
            if back6.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > back6.tStartRefresh + 10-frameTolerance:
                    # keep track of stop time/frame for later
                    back6.tStop = t  # not accounting for scr refresh
                    back6.tStopRefresh = tThisFlipGlobal  # on global time
                    back6.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'back6.stopped')
                    # update status
                    back6.status = FINISHED
                    back6.setAutoDraw(False)
            
            # *Quest* updates
            
            # if Quest is starting this frame...
            if Quest.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                Quest.frameNStart = frameN  # exact frame index
                Quest.tStart = t  # local t and not account for scr refresh
                Quest.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Quest, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Quest.started')
                # update status
                Quest.status = STARTED
                Quest.setAutoDraw(True)
            
            # if Quest is active this frame...
            if Quest.status == STARTED:
                # update params
                pass
            
            # if Quest is stopping this frame...
            if Quest.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > Quest.tStartRefresh + 10-frameTolerance:
                    # keep track of stop time/frame for later
                    Quest.tStop = t  # not accounting for scr refresh
                    Quest.tStopRefresh = tThisFlipGlobal  # on global time
                    Quest.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'Quest.stopped')
                    # update status
                    Quest.status = FINISHED
                    Quest.setAutoDraw(False)
            
            # *RatingResp* updates
            
            # if RatingResp is starting this frame...
            if RatingResp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                RatingResp.frameNStart = frameN  # exact frame index
                RatingResp.tStart = t  # local t and not account for scr refresh
                RatingResp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(RatingResp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'RatingResp.started')
                # update status
                RatingResp.status = STARTED
                RatingResp.setAutoDraw(True)
            
            # if RatingResp is active this frame...
            if RatingResp.status == STARTED:
                # update params
                pass
            
            # if RatingResp is stopping this frame...
            if RatingResp.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > RatingResp.tStartRefresh + 10-frameTolerance:
                    # keep track of stop time/frame for later
                    RatingResp.tStop = t  # not accounting for scr refresh
                    RatingResp.tStopRefresh = tThisFlipGlobal  # on global time
                    RatingResp.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'RatingResp.stopped')
                    # update status
                    RatingResp.status = FINISHED
                    RatingResp.setAutoDraw(False)
            
            # Check RatingResp for response to end Routine
            if RatingResp.getRating() is not None and RatingResp.status == STARTED:
                continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=Response,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                Response.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if Response.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in Response.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Response" ---
        for thisComponent in Response.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for Response
        Response.tStop = globalClock.getTime(format='float')
        Response.tStopRefresh = tThisFlipGlobal
        thisExp.addData('Response.stopped', Response.tStop)
        PractTrial.addData('RatingResp.response', RatingResp.getRating())
        PractTrial.addData('RatingResp.rt', RatingResp.getRT())
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if Response.maxDurationReached:
            routineTimer.addTime(-Response.maxDuration)
        elif Response.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-10.000000)
        
        # --- Prepare to start Routine "Fc" ---
        # create an object to store info about Routine Fc
        Fc = data.Routine(
            name='Fc',
            components=[back2, FixCross],
        )
        Fc.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for Fc
        Fc.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        Fc.tStart = globalClock.getTime(format='float')
        Fc.status = STARTED
        thisExp.addData('Fc.started', Fc.tStart)
        Fc.maxDuration = None
        # keep track of which components have finished
        FcComponents = Fc.components
        for thisComponent in Fc.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "Fc" ---
        thisExp.currentRoutine = Fc
        Fc.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 1.0:
            # if trial has changed, end Routine now
            if hasattr(thisPractTrial, 'status') and thisPractTrial.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *back2* updates
            
            # if back2 is starting this frame...
            if back2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                back2.frameNStart = frameN  # exact frame index
                back2.tStart = t  # local t and not account for scr refresh
                back2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(back2, 'tStartRefresh')  # time at next scr refresh
                # update status
                back2.status = STARTED
                back2.setAutoDraw(True)
            
            # if back2 is active this frame...
            if back2.status == STARTED:
                # update params
                pass
            
            # if back2 is stopping this frame...
            if back2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > back2.tStartRefresh + 1-frameTolerance:
                    # keep track of stop time/frame for later
                    back2.tStop = t  # not accounting for scr refresh
                    back2.tStopRefresh = tThisFlipGlobal  # on global time
                    back2.frameNStop = frameN  # exact frame index
                    # update status
                    back2.status = FINISHED
                    back2.setAutoDraw(False)
            
            # *FixCross* updates
            
            # if FixCross is starting this frame...
            if FixCross.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                FixCross.frameNStart = frameN  # exact frame index
                FixCross.tStart = t  # local t and not account for scr refresh
                FixCross.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(FixCross, 'tStartRefresh')  # time at next scr refresh
                # update status
                FixCross.status = STARTED
                FixCross.setAutoDraw(True)
            
            # if FixCross is active this frame...
            if FixCross.status == STARTED:
                # update params
                pass
            
            # if FixCross is stopping this frame...
            if FixCross.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > FixCross.tStartRefresh + 1.0-frameTolerance:
                    # keep track of stop time/frame for later
                    FixCross.tStop = t  # not accounting for scr refresh
                    FixCross.tStopRefresh = tThisFlipGlobal  # on global time
                    FixCross.frameNStop = frameN  # exact frame index
                    # update status
                    FixCross.status = FINISHED
                    FixCross.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=Fc,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                Fc.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if Fc.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in Fc.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Fc" ---
        for thisComponent in Fc.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for Fc
        Fc.tStop = globalClock.getTime(format='float')
        Fc.tStopRefresh = tThisFlipGlobal
        thisExp.addData('Fc.stopped', Fc.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if Fc.maxDurationReached:
            routineTimer.addTime(-Fc.maxDuration)
        elif Fc.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.000000)
        # mark thisPractTrial as finished
        if hasattr(thisPractTrial, 'status'):
            thisPractTrial.status = FINISHED
        # if awaiting a pause, pause now
        if PractTrial.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            PractTrial.status = STARTED
    # completed 1.0 repeats of 'PractTrial'
    PractTrial.status = FINISHED
    
    
    # --- Prepare to start Routine "instruction" ---
    # create an object to store info about Routine instruction
    instruction = data.Routine(
        name='instruction',
        components=[background, PressText, Resp1Space],
    )
    instruction.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for Resp1Space
    Resp1Space.keys = []
    Resp1Space.rt = []
    _Resp1Space_allKeys = []
    # store start times for instruction
    instruction.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    instruction.tStart = globalClock.getTime(format='float')
    instruction.status = STARTED
    thisExp.addData('instruction.started', instruction.tStart)
    instruction.maxDuration = None
    # keep track of which components have finished
    instructionComponents = instruction.components
    for thisComponent in instruction.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "instruction" ---
    thisExp.currentRoutine = instruction
    instruction.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *background* updates
        
        # if background is starting this frame...
        if background.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            background.frameNStart = frameN  # exact frame index
            background.tStart = t  # local t and not account for scr refresh
            background.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(background, 'tStartRefresh')  # time at next scr refresh
            # update status
            background.status = STARTED
            background.setAutoDraw(True)
        
        # if background is active this frame...
        if background.status == STARTED:
            # update params
            pass
        
        # *PressText* updates
        
        # if PressText is starting this frame...
        if PressText.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            PressText.frameNStart = frameN  # exact frame index
            PressText.tStart = t  # local t and not account for scr refresh
            PressText.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(PressText, 'tStartRefresh')  # time at next scr refresh
            # update status
            PressText.status = STARTED
            PressText.setAutoDraw(True)
        
        # if PressText is active this frame...
        if PressText.status == STARTED:
            # update params
            pass
        
        # *Resp1Space* updates
        waitOnFlip = False
        
        # if Resp1Space is starting this frame...
        if Resp1Space.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            Resp1Space.frameNStart = frameN  # exact frame index
            Resp1Space.tStart = t  # local t and not account for scr refresh
            Resp1Space.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Resp1Space, 'tStartRefresh')  # time at next scr refresh
            # update status
            Resp1Space.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(Resp1Space.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(Resp1Space.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if Resp1Space.status == STARTED and not waitOnFlip:
            theseKeys = Resp1Space.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _Resp1Space_allKeys.extend(theseKeys)
            if len(_Resp1Space_allKeys):
                Resp1Space.keys = _Resp1Space_allKeys[-1].name  # just the last key pressed
                Resp1Space.rt = _Resp1Space_allKeys[-1].rt
                Resp1Space.duration = _Resp1Space_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=instruction,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            instruction.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if instruction.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in instruction.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instruction" ---
    for thisComponent in instruction.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for instruction
    instruction.tStop = globalClock.getTime(format='float')
    instruction.tStopRefresh = tThisFlipGlobal
    thisExp.addData('instruction.stopped', instruction.tStop)
    thisExp.nextEntry()
    # the Routine "instruction" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "Fc" ---
    # create an object to store info about Routine Fc
    Fc = data.Routine(
        name='Fc',
        components=[back2, FixCross],
    )
    Fc.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for Fc
    Fc.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    Fc.tStart = globalClock.getTime(format='float')
    Fc.status = STARTED
    thisExp.addData('Fc.started', Fc.tStart)
    Fc.maxDuration = None
    # keep track of which components have finished
    FcComponents = Fc.components
    for thisComponent in Fc.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Fc" ---
    thisExp.currentRoutine = Fc
    Fc.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 1.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *back2* updates
        
        # if back2 is starting this frame...
        if back2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            back2.frameNStart = frameN  # exact frame index
            back2.tStart = t  # local t and not account for scr refresh
            back2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(back2, 'tStartRefresh')  # time at next scr refresh
            # update status
            back2.status = STARTED
            back2.setAutoDraw(True)
        
        # if back2 is active this frame...
        if back2.status == STARTED:
            # update params
            pass
        
        # if back2 is stopping this frame...
        if back2.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > back2.tStartRefresh + 1-frameTolerance:
                # keep track of stop time/frame for later
                back2.tStop = t  # not accounting for scr refresh
                back2.tStopRefresh = tThisFlipGlobal  # on global time
                back2.frameNStop = frameN  # exact frame index
                # update status
                back2.status = FINISHED
                back2.setAutoDraw(False)
        
        # *FixCross* updates
        
        # if FixCross is starting this frame...
        if FixCross.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            FixCross.frameNStart = frameN  # exact frame index
            FixCross.tStart = t  # local t and not account for scr refresh
            FixCross.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(FixCross, 'tStartRefresh')  # time at next scr refresh
            # update status
            FixCross.status = STARTED
            FixCross.setAutoDraw(True)
        
        # if FixCross is active this frame...
        if FixCross.status == STARTED:
            # update params
            pass
        
        # if FixCross is stopping this frame...
        if FixCross.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > FixCross.tStartRefresh + 1.0-frameTolerance:
                # keep track of stop time/frame for later
                FixCross.tStop = t  # not accounting for scr refresh
                FixCross.tStopRefresh = tThisFlipGlobal  # on global time
                FixCross.frameNStop = frameN  # exact frame index
                # update status
                FixCross.status = FINISHED
                FixCross.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=Fc,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            Fc.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if Fc.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in Fc.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Fc" ---
    for thisComponent in Fc.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for Fc
    Fc.tStop = globalClock.getTime(format='float')
    Fc.tStopRefresh = tThisFlipGlobal
    thisExp.addData('Fc.stopped', Fc.tStop)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if Fc.maxDurationReached:
        routineTimer.addTime(-Fc.maxDuration)
    elif Fc.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-1.000000)
    thisExp.nextEntry()
    
    # set up handler to look after randomisation of conditions etc
    Tr1 = data.TrialHandler2(
        name='Tr1',
        nReps=1.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('before.xlsx'), 
        seed=None, 
        isTrials=True, 
    )
    thisExp.addLoop(Tr1)  # add the loop to the experiment
    thisTr1 = Tr1.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTr1.rgb)
    if thisTr1 != None:
        for paramName in thisTr1:
            globals()[paramName] = thisTr1[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisTr1 in Tr1:
        Tr1.status = STARTED
        if hasattr(thisTr1, 'status'):
            thisTr1.status = STARTED
        currentLoop = Tr1
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisTr1.rgb)
        if thisTr1 != None:
            for paramName in thisTr1:
                globals()[paramName] = thisTr1[paramName]
        
        # --- Prepare to start Routine "taskT1" ---
        # create an object to store info about Routine taskT1
        taskT1 = data.Routine(
            name='taskT1',
            components=[back3, OtherText, Pic, Vc],
        )
        taskT1.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        OtherText.setText('The Other Person is listening to...')
        Pic.setImage(image)
        Vc.setSound(vc, secs=duration, hamming=True)
        Vc.setVolume(1.0, log=False)
        Vc.seek(0)
        # store start times for taskT1
        taskT1.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        taskT1.tStart = globalClock.getTime(format='float')
        taskT1.status = STARTED
        thisExp.addData('taskT1.started', taskT1.tStart)
        taskT1.maxDuration = None
        # keep track of which components have finished
        taskT1Components = taskT1.components
        for thisComponent in taskT1.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "taskT1" ---
        thisExp.currentRoutine = taskT1
        taskT1.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisTr1, 'status') and thisTr1.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *back3* updates
            
            # if back3 is starting this frame...
            if back3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                back3.frameNStart = frameN  # exact frame index
                back3.tStart = t  # local t and not account for scr refresh
                back3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(back3, 'tStartRefresh')  # time at next scr refresh
                # update status
                back3.status = STARTED
                back3.setAutoDraw(True)
            
            # if back3 is active this frame...
            if back3.status == STARTED:
                # update params
                pass
            
            # if back3 is stopping this frame...
            if back3.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > back3.tStartRefresh + duration-frameTolerance:
                    # keep track of stop time/frame for later
                    back3.tStop = t  # not accounting for scr refresh
                    back3.tStopRefresh = tThisFlipGlobal  # on global time
                    back3.frameNStop = frameN  # exact frame index
                    # update status
                    back3.status = FINISHED
                    back3.setAutoDraw(False)
            
            # *OtherText* updates
            
            # if OtherText is starting this frame...
            if OtherText.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                OtherText.frameNStart = frameN  # exact frame index
                OtherText.tStart = t  # local t and not account for scr refresh
                OtherText.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(OtherText, 'tStartRefresh')  # time at next scr refresh
                # update status
                OtherText.status = STARTED
                OtherText.setAutoDraw(True)
            
            # if OtherText is active this frame...
            if OtherText.status == STARTED:
                # update params
                pass
            
            # if OtherText is stopping this frame...
            if OtherText.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > OtherText.tStartRefresh + duration-frameTolerance:
                    # keep track of stop time/frame for later
                    OtherText.tStop = t  # not accounting for scr refresh
                    OtherText.tStopRefresh = tThisFlipGlobal  # on global time
                    OtherText.frameNStop = frameN  # exact frame index
                    # update status
                    OtherText.status = FINISHED
                    OtherText.setAutoDraw(False)
            
            # *Pic* updates
            
            # if Pic is starting this frame...
            if Pic.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                Pic.frameNStart = frameN  # exact frame index
                Pic.tStart = t  # local t and not account for scr refresh
                Pic.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Pic, 'tStartRefresh')  # time at next scr refresh
                # update status
                Pic.status = STARTED
                Pic.setAutoDraw(True)
            
            # if Pic is active this frame...
            if Pic.status == STARTED:
                # update params
                pass
            
            # if Pic is stopping this frame...
            if Pic.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > Pic.tStartRefresh + duration-frameTolerance:
                    # keep track of stop time/frame for later
                    Pic.tStop = t  # not accounting for scr refresh
                    Pic.tStopRefresh = tThisFlipGlobal  # on global time
                    Pic.frameNStop = frameN  # exact frame index
                    # update status
                    Pic.status = FINISHED
                    Pic.setAutoDraw(False)
            
            # *Vc* updates
            
            # if Vc is starting this frame...
            if Vc.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                Vc.frameNStart = frameN  # exact frame index
                Vc.tStart = t  # local t and not account for scr refresh
                Vc.tStartRefresh = tThisFlipGlobal  # on global time
                # update status
                Vc.status = STARTED
                Vc.play(when=win)  # sync with win flip
            
            # if Vc is stopping this frame...
            if Vc.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > Vc.tStartRefresh + duration-frameTolerance or Vc.isFinished:
                    # keep track of stop time/frame for later
                    Vc.tStop = t  # not accounting for scr refresh
                    Vc.tStopRefresh = tThisFlipGlobal  # on global time
                    Vc.frameNStop = frameN  # exact frame index
                    # update status
                    Vc.status = FINISHED
                    Vc.stop()
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=taskT1,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                taskT1.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if taskT1.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in taskT1.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "taskT1" ---
        for thisComponent in taskT1.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for taskT1
        taskT1.tStop = globalClock.getTime(format='float')
        taskT1.tStopRefresh = tThisFlipGlobal
        thisExp.addData('taskT1.stopped', taskT1.tStop)
        Vc.pause()  # ensure sound has stopped at end of Routine
        # the Routine "taskT1" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "Response" ---
        # create an object to store info about Routine Response
        Response = data.Routine(
            name='Response',
            components=[back6, Quest, RatingResp],
        )
        Response.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        Quest.setText(text)
        RatingResp.reset()
        # store start times for Response
        Response.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        Response.tStart = globalClock.getTime(format='float')
        Response.status = STARTED
        thisExp.addData('Response.started', Response.tStart)
        Response.maxDuration = None
        # keep track of which components have finished
        ResponseComponents = Response.components
        for thisComponent in Response.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "Response" ---
        thisExp.currentRoutine = Response
        Response.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 10.0:
            # if trial has changed, end Routine now
            if hasattr(thisTr1, 'status') and thisTr1.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *back6* updates
            
            # if back6 is starting this frame...
            if back6.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                back6.frameNStart = frameN  # exact frame index
                back6.tStart = t  # local t and not account for scr refresh
                back6.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(back6, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'back6.started')
                # update status
                back6.status = STARTED
                back6.setAutoDraw(True)
            
            # if back6 is active this frame...
            if back6.status == STARTED:
                # update params
                pass
            
            # if back6 is stopping this frame...
            if back6.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > back6.tStartRefresh + 10-frameTolerance:
                    # keep track of stop time/frame for later
                    back6.tStop = t  # not accounting for scr refresh
                    back6.tStopRefresh = tThisFlipGlobal  # on global time
                    back6.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'back6.stopped')
                    # update status
                    back6.status = FINISHED
                    back6.setAutoDraw(False)
            
            # *Quest* updates
            
            # if Quest is starting this frame...
            if Quest.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                Quest.frameNStart = frameN  # exact frame index
                Quest.tStart = t  # local t and not account for scr refresh
                Quest.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Quest, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Quest.started')
                # update status
                Quest.status = STARTED
                Quest.setAutoDraw(True)
            
            # if Quest is active this frame...
            if Quest.status == STARTED:
                # update params
                pass
            
            # if Quest is stopping this frame...
            if Quest.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > Quest.tStartRefresh + 10-frameTolerance:
                    # keep track of stop time/frame for later
                    Quest.tStop = t  # not accounting for scr refresh
                    Quest.tStopRefresh = tThisFlipGlobal  # on global time
                    Quest.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'Quest.stopped')
                    # update status
                    Quest.status = FINISHED
                    Quest.setAutoDraw(False)
            
            # *RatingResp* updates
            
            # if RatingResp is starting this frame...
            if RatingResp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                RatingResp.frameNStart = frameN  # exact frame index
                RatingResp.tStart = t  # local t and not account for scr refresh
                RatingResp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(RatingResp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'RatingResp.started')
                # update status
                RatingResp.status = STARTED
                RatingResp.setAutoDraw(True)
            
            # if RatingResp is active this frame...
            if RatingResp.status == STARTED:
                # update params
                pass
            
            # if RatingResp is stopping this frame...
            if RatingResp.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > RatingResp.tStartRefresh + 10-frameTolerance:
                    # keep track of stop time/frame for later
                    RatingResp.tStop = t  # not accounting for scr refresh
                    RatingResp.tStopRefresh = tThisFlipGlobal  # on global time
                    RatingResp.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'RatingResp.stopped')
                    # update status
                    RatingResp.status = FINISHED
                    RatingResp.setAutoDraw(False)
            
            # Check RatingResp for response to end Routine
            if RatingResp.getRating() is not None and RatingResp.status == STARTED:
                continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=Response,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                Response.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if Response.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in Response.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Response" ---
        for thisComponent in Response.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for Response
        Response.tStop = globalClock.getTime(format='float')
        Response.tStopRefresh = tThisFlipGlobal
        thisExp.addData('Response.stopped', Response.tStop)
        Tr1.addData('RatingResp.response', RatingResp.getRating())
        Tr1.addData('RatingResp.rt', RatingResp.getRT())
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if Response.maxDurationReached:
            routineTimer.addTime(-Response.maxDuration)
        elif Response.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-10.000000)
        
        # --- Prepare to start Routine "Fc" ---
        # create an object to store info about Routine Fc
        Fc = data.Routine(
            name='Fc',
            components=[back2, FixCross],
        )
        Fc.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for Fc
        Fc.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        Fc.tStart = globalClock.getTime(format='float')
        Fc.status = STARTED
        thisExp.addData('Fc.started', Fc.tStart)
        Fc.maxDuration = None
        # keep track of which components have finished
        FcComponents = Fc.components
        for thisComponent in Fc.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "Fc" ---
        thisExp.currentRoutine = Fc
        Fc.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 1.0:
            # if trial has changed, end Routine now
            if hasattr(thisTr1, 'status') and thisTr1.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *back2* updates
            
            # if back2 is starting this frame...
            if back2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                back2.frameNStart = frameN  # exact frame index
                back2.tStart = t  # local t and not account for scr refresh
                back2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(back2, 'tStartRefresh')  # time at next scr refresh
                # update status
                back2.status = STARTED
                back2.setAutoDraw(True)
            
            # if back2 is active this frame...
            if back2.status == STARTED:
                # update params
                pass
            
            # if back2 is stopping this frame...
            if back2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > back2.tStartRefresh + 1-frameTolerance:
                    # keep track of stop time/frame for later
                    back2.tStop = t  # not accounting for scr refresh
                    back2.tStopRefresh = tThisFlipGlobal  # on global time
                    back2.frameNStop = frameN  # exact frame index
                    # update status
                    back2.status = FINISHED
                    back2.setAutoDraw(False)
            
            # *FixCross* updates
            
            # if FixCross is starting this frame...
            if FixCross.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                FixCross.frameNStart = frameN  # exact frame index
                FixCross.tStart = t  # local t and not account for scr refresh
                FixCross.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(FixCross, 'tStartRefresh')  # time at next scr refresh
                # update status
                FixCross.status = STARTED
                FixCross.setAutoDraw(True)
            
            # if FixCross is active this frame...
            if FixCross.status == STARTED:
                # update params
                pass
            
            # if FixCross is stopping this frame...
            if FixCross.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > FixCross.tStartRefresh + 1.0-frameTolerance:
                    # keep track of stop time/frame for later
                    FixCross.tStop = t  # not accounting for scr refresh
                    FixCross.tStopRefresh = tThisFlipGlobal  # on global time
                    FixCross.frameNStop = frameN  # exact frame index
                    # update status
                    FixCross.status = FINISHED
                    FixCross.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=Fc,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                Fc.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if Fc.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in Fc.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Fc" ---
        for thisComponent in Fc.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for Fc
        Fc.tStop = globalClock.getTime(format='float')
        Fc.tStopRefresh = tThisFlipGlobal
        thisExp.addData('Fc.stopped', Fc.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if Fc.maxDurationReached:
            routineTimer.addTime(-Fc.maxDuration)
        elif Fc.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.000000)
        # mark thisTr1 as finished
        if hasattr(thisTr1, 'status'):
            thisTr1.status = FINISHED
        # if awaiting a pause, pause now
        if Tr1.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            Tr1.status = STARTED
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'Tr1'
    Tr1.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "BreakRoutine" ---
    # create an object to store info about Routine BreakRoutine
    BreakRoutine = data.Routine(
        name='BreakRoutine',
        components=[back5, BreakText, Resp2Space],
    )
    BreakRoutine.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for Resp2Space
    Resp2Space.keys = []
    Resp2Space.rt = []
    _Resp2Space_allKeys = []
    # store start times for BreakRoutine
    BreakRoutine.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    BreakRoutine.tStart = globalClock.getTime(format='float')
    BreakRoutine.status = STARTED
    thisExp.addData('BreakRoutine.started', BreakRoutine.tStart)
    BreakRoutine.maxDuration = None
    # keep track of which components have finished
    BreakRoutineComponents = BreakRoutine.components
    for thisComponent in BreakRoutine.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "BreakRoutine" ---
    thisExp.currentRoutine = BreakRoutine
    BreakRoutine.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *back5* updates
        
        # if back5 is starting this frame...
        if back5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            back5.frameNStart = frameN  # exact frame index
            back5.tStart = t  # local t and not account for scr refresh
            back5.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(back5, 'tStartRefresh')  # time at next scr refresh
            # update status
            back5.status = STARTED
            back5.setAutoDraw(True)
        
        # if back5 is active this frame...
        if back5.status == STARTED:
            # update params
            pass
        
        # *BreakText* updates
        
        # if BreakText is starting this frame...
        if BreakText.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            BreakText.frameNStart = frameN  # exact frame index
            BreakText.tStart = t  # local t and not account for scr refresh
            BreakText.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(BreakText, 'tStartRefresh')  # time at next scr refresh
            # update status
            BreakText.status = STARTED
            BreakText.setAutoDraw(True)
        
        # if BreakText is active this frame...
        if BreakText.status == STARTED:
            # update params
            pass
        
        # *Resp2Space* updates
        waitOnFlip = False
        
        # if Resp2Space is starting this frame...
        if Resp2Space.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            Resp2Space.frameNStart = frameN  # exact frame index
            Resp2Space.tStart = t  # local t and not account for scr refresh
            Resp2Space.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Resp2Space, 'tStartRefresh')  # time at next scr refresh
            # update status
            Resp2Space.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(Resp2Space.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(Resp2Space.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if Resp2Space.status == STARTED and not waitOnFlip:
            theseKeys = Resp2Space.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _Resp2Space_allKeys.extend(theseKeys)
            if len(_Resp2Space_allKeys):
                Resp2Space.keys = _Resp2Space_allKeys[-1].name  # just the last key pressed
                Resp2Space.rt = _Resp2Space_allKeys[-1].rt
                Resp2Space.duration = _Resp2Space_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=BreakRoutine,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            BreakRoutine.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if BreakRoutine.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in BreakRoutine.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "BreakRoutine" ---
    for thisComponent in BreakRoutine.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for BreakRoutine
    BreakRoutine.tStop = globalClock.getTime(format='float')
    BreakRoutine.tStopRefresh = tThisFlipGlobal
    thisExp.addData('BreakRoutine.stopped', BreakRoutine.tStop)
    thisExp.nextEntry()
    # the Routine "BreakRoutine" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "Fc" ---
    # create an object to store info about Routine Fc
    Fc = data.Routine(
        name='Fc',
        components=[back2, FixCross],
    )
    Fc.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for Fc
    Fc.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    Fc.tStart = globalClock.getTime(format='float')
    Fc.status = STARTED
    thisExp.addData('Fc.started', Fc.tStart)
    Fc.maxDuration = None
    # keep track of which components have finished
    FcComponents = Fc.components
    for thisComponent in Fc.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Fc" ---
    thisExp.currentRoutine = Fc
    Fc.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 1.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *back2* updates
        
        # if back2 is starting this frame...
        if back2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            back2.frameNStart = frameN  # exact frame index
            back2.tStart = t  # local t and not account for scr refresh
            back2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(back2, 'tStartRefresh')  # time at next scr refresh
            # update status
            back2.status = STARTED
            back2.setAutoDraw(True)
        
        # if back2 is active this frame...
        if back2.status == STARTED:
            # update params
            pass
        
        # if back2 is stopping this frame...
        if back2.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > back2.tStartRefresh + 1-frameTolerance:
                # keep track of stop time/frame for later
                back2.tStop = t  # not accounting for scr refresh
                back2.tStopRefresh = tThisFlipGlobal  # on global time
                back2.frameNStop = frameN  # exact frame index
                # update status
                back2.status = FINISHED
                back2.setAutoDraw(False)
        
        # *FixCross* updates
        
        # if FixCross is starting this frame...
        if FixCross.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            FixCross.frameNStart = frameN  # exact frame index
            FixCross.tStart = t  # local t and not account for scr refresh
            FixCross.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(FixCross, 'tStartRefresh')  # time at next scr refresh
            # update status
            FixCross.status = STARTED
            FixCross.setAutoDraw(True)
        
        # if FixCross is active this frame...
        if FixCross.status == STARTED:
            # update params
            pass
        
        # if FixCross is stopping this frame...
        if FixCross.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > FixCross.tStartRefresh + 1.0-frameTolerance:
                # keep track of stop time/frame for later
                FixCross.tStop = t  # not accounting for scr refresh
                FixCross.tStopRefresh = tThisFlipGlobal  # on global time
                FixCross.frameNStop = frameN  # exact frame index
                # update status
                FixCross.status = FINISHED
                FixCross.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=Fc,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            Fc.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if Fc.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in Fc.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Fc" ---
    for thisComponent in Fc.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for Fc
    Fc.tStop = globalClock.getTime(format='float')
    Fc.tStopRefresh = tThisFlipGlobal
    thisExp.addData('Fc.stopped', Fc.tStop)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if Fc.maxDurationReached:
        routineTimer.addTime(-Fc.maxDuration)
    elif Fc.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-1.000000)
    thisExp.nextEntry()
    
    # set up handler to look after randomisation of conditions etc
    Tr2 = data.TrialHandler2(
        name='Tr2',
        nReps=1.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('after.xlsx'), 
        seed=None, 
        isTrials=True, 
    )
    thisExp.addLoop(Tr2)  # add the loop to the experiment
    thisTr2 = Tr2.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTr2.rgb)
    if thisTr2 != None:
        for paramName in thisTr2:
            globals()[paramName] = thisTr2[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisTr2 in Tr2:
        Tr2.status = STARTED
        if hasattr(thisTr2, 'status'):
            thisTr2.status = STARTED
        currentLoop = Tr2
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisTr2.rgb)
        if thisTr2 != None:
            for paramName in thisTr2:
                globals()[paramName] = thisTr2[paramName]
        
        # --- Prepare to start Routine "taskT1" ---
        # create an object to store info about Routine taskT1
        taskT1 = data.Routine(
            name='taskT1',
            components=[back3, OtherText, Pic, Vc],
        )
        taskT1.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        OtherText.setText('The Other Person is listening to...')
        Pic.setImage(image)
        Vc.setSound(vc, secs=duration, hamming=True)
        Vc.setVolume(1.0, log=False)
        Vc.seek(0)
        # store start times for taskT1
        taskT1.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        taskT1.tStart = globalClock.getTime(format='float')
        taskT1.status = STARTED
        thisExp.addData('taskT1.started', taskT1.tStart)
        taskT1.maxDuration = None
        # keep track of which components have finished
        taskT1Components = taskT1.components
        for thisComponent in taskT1.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "taskT1" ---
        thisExp.currentRoutine = taskT1
        taskT1.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisTr2, 'status') and thisTr2.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *back3* updates
            
            # if back3 is starting this frame...
            if back3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                back3.frameNStart = frameN  # exact frame index
                back3.tStart = t  # local t and not account for scr refresh
                back3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(back3, 'tStartRefresh')  # time at next scr refresh
                # update status
                back3.status = STARTED
                back3.setAutoDraw(True)
            
            # if back3 is active this frame...
            if back3.status == STARTED:
                # update params
                pass
            
            # if back3 is stopping this frame...
            if back3.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > back3.tStartRefresh + duration-frameTolerance:
                    # keep track of stop time/frame for later
                    back3.tStop = t  # not accounting for scr refresh
                    back3.tStopRefresh = tThisFlipGlobal  # on global time
                    back3.frameNStop = frameN  # exact frame index
                    # update status
                    back3.status = FINISHED
                    back3.setAutoDraw(False)
            
            # *OtherText* updates
            
            # if OtherText is starting this frame...
            if OtherText.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                OtherText.frameNStart = frameN  # exact frame index
                OtherText.tStart = t  # local t and not account for scr refresh
                OtherText.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(OtherText, 'tStartRefresh')  # time at next scr refresh
                # update status
                OtherText.status = STARTED
                OtherText.setAutoDraw(True)
            
            # if OtherText is active this frame...
            if OtherText.status == STARTED:
                # update params
                pass
            
            # if OtherText is stopping this frame...
            if OtherText.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > OtherText.tStartRefresh + duration-frameTolerance:
                    # keep track of stop time/frame for later
                    OtherText.tStop = t  # not accounting for scr refresh
                    OtherText.tStopRefresh = tThisFlipGlobal  # on global time
                    OtherText.frameNStop = frameN  # exact frame index
                    # update status
                    OtherText.status = FINISHED
                    OtherText.setAutoDraw(False)
            
            # *Pic* updates
            
            # if Pic is starting this frame...
            if Pic.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                Pic.frameNStart = frameN  # exact frame index
                Pic.tStart = t  # local t and not account for scr refresh
                Pic.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Pic, 'tStartRefresh')  # time at next scr refresh
                # update status
                Pic.status = STARTED
                Pic.setAutoDraw(True)
            
            # if Pic is active this frame...
            if Pic.status == STARTED:
                # update params
                pass
            
            # if Pic is stopping this frame...
            if Pic.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > Pic.tStartRefresh + duration-frameTolerance:
                    # keep track of stop time/frame for later
                    Pic.tStop = t  # not accounting for scr refresh
                    Pic.tStopRefresh = tThisFlipGlobal  # on global time
                    Pic.frameNStop = frameN  # exact frame index
                    # update status
                    Pic.status = FINISHED
                    Pic.setAutoDraw(False)
            
            # *Vc* updates
            
            # if Vc is starting this frame...
            if Vc.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                Vc.frameNStart = frameN  # exact frame index
                Vc.tStart = t  # local t and not account for scr refresh
                Vc.tStartRefresh = tThisFlipGlobal  # on global time
                # update status
                Vc.status = STARTED
                Vc.play(when=win)  # sync with win flip
            
            # if Vc is stopping this frame...
            if Vc.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > Vc.tStartRefresh + duration-frameTolerance or Vc.isFinished:
                    # keep track of stop time/frame for later
                    Vc.tStop = t  # not accounting for scr refresh
                    Vc.tStopRefresh = tThisFlipGlobal  # on global time
                    Vc.frameNStop = frameN  # exact frame index
                    # update status
                    Vc.status = FINISHED
                    Vc.stop()
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=taskT1,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                taskT1.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if taskT1.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in taskT1.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "taskT1" ---
        for thisComponent in taskT1.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for taskT1
        taskT1.tStop = globalClock.getTime(format='float')
        taskT1.tStopRefresh = tThisFlipGlobal
        thisExp.addData('taskT1.stopped', taskT1.tStop)
        Vc.pause()  # ensure sound has stopped at end of Routine
        # the Routine "taskT1" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "Response" ---
        # create an object to store info about Routine Response
        Response = data.Routine(
            name='Response',
            components=[back6, Quest, RatingResp],
        )
        Response.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        Quest.setText(text)
        RatingResp.reset()
        # store start times for Response
        Response.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        Response.tStart = globalClock.getTime(format='float')
        Response.status = STARTED
        thisExp.addData('Response.started', Response.tStart)
        Response.maxDuration = None
        # keep track of which components have finished
        ResponseComponents = Response.components
        for thisComponent in Response.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "Response" ---
        thisExp.currentRoutine = Response
        Response.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 10.0:
            # if trial has changed, end Routine now
            if hasattr(thisTr2, 'status') and thisTr2.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *back6* updates
            
            # if back6 is starting this frame...
            if back6.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                back6.frameNStart = frameN  # exact frame index
                back6.tStart = t  # local t and not account for scr refresh
                back6.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(back6, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'back6.started')
                # update status
                back6.status = STARTED
                back6.setAutoDraw(True)
            
            # if back6 is active this frame...
            if back6.status == STARTED:
                # update params
                pass
            
            # if back6 is stopping this frame...
            if back6.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > back6.tStartRefresh + 10-frameTolerance:
                    # keep track of stop time/frame for later
                    back6.tStop = t  # not accounting for scr refresh
                    back6.tStopRefresh = tThisFlipGlobal  # on global time
                    back6.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'back6.stopped')
                    # update status
                    back6.status = FINISHED
                    back6.setAutoDraw(False)
            
            # *Quest* updates
            
            # if Quest is starting this frame...
            if Quest.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                Quest.frameNStart = frameN  # exact frame index
                Quest.tStart = t  # local t and not account for scr refresh
                Quest.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Quest, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Quest.started')
                # update status
                Quest.status = STARTED
                Quest.setAutoDraw(True)
            
            # if Quest is active this frame...
            if Quest.status == STARTED:
                # update params
                pass
            
            # if Quest is stopping this frame...
            if Quest.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > Quest.tStartRefresh + 10-frameTolerance:
                    # keep track of stop time/frame for later
                    Quest.tStop = t  # not accounting for scr refresh
                    Quest.tStopRefresh = tThisFlipGlobal  # on global time
                    Quest.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'Quest.stopped')
                    # update status
                    Quest.status = FINISHED
                    Quest.setAutoDraw(False)
            
            # *RatingResp* updates
            
            # if RatingResp is starting this frame...
            if RatingResp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                RatingResp.frameNStart = frameN  # exact frame index
                RatingResp.tStart = t  # local t and not account for scr refresh
                RatingResp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(RatingResp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'RatingResp.started')
                # update status
                RatingResp.status = STARTED
                RatingResp.setAutoDraw(True)
            
            # if RatingResp is active this frame...
            if RatingResp.status == STARTED:
                # update params
                pass
            
            # if RatingResp is stopping this frame...
            if RatingResp.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > RatingResp.tStartRefresh + 10-frameTolerance:
                    # keep track of stop time/frame for later
                    RatingResp.tStop = t  # not accounting for scr refresh
                    RatingResp.tStopRefresh = tThisFlipGlobal  # on global time
                    RatingResp.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'RatingResp.stopped')
                    # update status
                    RatingResp.status = FINISHED
                    RatingResp.setAutoDraw(False)
            
            # Check RatingResp for response to end Routine
            if RatingResp.getRating() is not None and RatingResp.status == STARTED:
                continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=Response,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                Response.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if Response.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in Response.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Response" ---
        for thisComponent in Response.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for Response
        Response.tStop = globalClock.getTime(format='float')
        Response.tStopRefresh = tThisFlipGlobal
        thisExp.addData('Response.stopped', Response.tStop)
        Tr2.addData('RatingResp.response', RatingResp.getRating())
        Tr2.addData('RatingResp.rt', RatingResp.getRT())
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if Response.maxDurationReached:
            routineTimer.addTime(-Response.maxDuration)
        elif Response.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-10.000000)
        
        # --- Prepare to start Routine "Fc" ---
        # create an object to store info about Routine Fc
        Fc = data.Routine(
            name='Fc',
            components=[back2, FixCross],
        )
        Fc.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for Fc
        Fc.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        Fc.tStart = globalClock.getTime(format='float')
        Fc.status = STARTED
        thisExp.addData('Fc.started', Fc.tStart)
        Fc.maxDuration = None
        # keep track of which components have finished
        FcComponents = Fc.components
        for thisComponent in Fc.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "Fc" ---
        thisExp.currentRoutine = Fc
        Fc.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 1.0:
            # if trial has changed, end Routine now
            if hasattr(thisTr2, 'status') and thisTr2.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *back2* updates
            
            # if back2 is starting this frame...
            if back2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                back2.frameNStart = frameN  # exact frame index
                back2.tStart = t  # local t and not account for scr refresh
                back2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(back2, 'tStartRefresh')  # time at next scr refresh
                # update status
                back2.status = STARTED
                back2.setAutoDraw(True)
            
            # if back2 is active this frame...
            if back2.status == STARTED:
                # update params
                pass
            
            # if back2 is stopping this frame...
            if back2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > back2.tStartRefresh + 1-frameTolerance:
                    # keep track of stop time/frame for later
                    back2.tStop = t  # not accounting for scr refresh
                    back2.tStopRefresh = tThisFlipGlobal  # on global time
                    back2.frameNStop = frameN  # exact frame index
                    # update status
                    back2.status = FINISHED
                    back2.setAutoDraw(False)
            
            # *FixCross* updates
            
            # if FixCross is starting this frame...
            if FixCross.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                FixCross.frameNStart = frameN  # exact frame index
                FixCross.tStart = t  # local t and not account for scr refresh
                FixCross.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(FixCross, 'tStartRefresh')  # time at next scr refresh
                # update status
                FixCross.status = STARTED
                FixCross.setAutoDraw(True)
            
            # if FixCross is active this frame...
            if FixCross.status == STARTED:
                # update params
                pass
            
            # if FixCross is stopping this frame...
            if FixCross.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > FixCross.tStartRefresh + 1.0-frameTolerance:
                    # keep track of stop time/frame for later
                    FixCross.tStop = t  # not accounting for scr refresh
                    FixCross.tStopRefresh = tThisFlipGlobal  # on global time
                    FixCross.frameNStop = frameN  # exact frame index
                    # update status
                    FixCross.status = FINISHED
                    FixCross.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=Fc,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                Fc.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if Fc.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in Fc.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Fc" ---
        for thisComponent in Fc.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for Fc
        Fc.tStop = globalClock.getTime(format='float')
        Fc.tStopRefresh = tThisFlipGlobal
        thisExp.addData('Fc.stopped', Fc.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if Fc.maxDurationReached:
            routineTimer.addTime(-Fc.maxDuration)
        elif Fc.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.000000)
        # mark thisTr2 as finished
        if hasattr(thisTr2, 'status'):
            thisTr2.status = FINISHED
        # if awaiting a pause, pause now
        if Tr2.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            Tr2.status = STARTED
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'Tr2'
    Tr2.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "Finish" ---
    # create an object to store info about Routine Finish
    Finish = data.Routine(
        name='Finish',
        components=[back7, TnxParticip, Resp3Space],
    )
    Finish.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for Resp3Space
    Resp3Space.keys = []
    Resp3Space.rt = []
    _Resp3Space_allKeys = []
    # store start times for Finish
    Finish.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    Finish.tStart = globalClock.getTime(format='float')
    Finish.status = STARTED
    thisExp.addData('Finish.started', Finish.tStart)
    Finish.maxDuration = None
    # keep track of which components have finished
    FinishComponents = Finish.components
    for thisComponent in Finish.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Finish" ---
    thisExp.currentRoutine = Finish
    Finish.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *back7* updates
        
        # if back7 is starting this frame...
        if back7.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            back7.frameNStart = frameN  # exact frame index
            back7.tStart = t  # local t and not account for scr refresh
            back7.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(back7, 'tStartRefresh')  # time at next scr refresh
            # update status
            back7.status = STARTED
            back7.setAutoDraw(True)
        
        # if back7 is active this frame...
        if back7.status == STARTED:
            # update params
            pass
        
        # *TnxParticip* updates
        
        # if TnxParticip is starting this frame...
        if TnxParticip.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            TnxParticip.frameNStart = frameN  # exact frame index
            TnxParticip.tStart = t  # local t and not account for scr refresh
            TnxParticip.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(TnxParticip, 'tStartRefresh')  # time at next scr refresh
            # update status
            TnxParticip.status = STARTED
            TnxParticip.setAutoDraw(True)
        
        # if TnxParticip is active this frame...
        if TnxParticip.status == STARTED:
            # update params
            pass
        
        # *Resp3Space* updates
        waitOnFlip = False
        
        # if Resp3Space is starting this frame...
        if Resp3Space.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            Resp3Space.frameNStart = frameN  # exact frame index
            Resp3Space.tStart = t  # local t and not account for scr refresh
            Resp3Space.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Resp3Space, 'tStartRefresh')  # time at next scr refresh
            # update status
            Resp3Space.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(Resp3Space.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(Resp3Space.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if Resp3Space.status == STARTED and not waitOnFlip:
            theseKeys = Resp3Space.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _Resp3Space_allKeys.extend(theseKeys)
            if len(_Resp3Space_allKeys):
                Resp3Space.keys = _Resp3Space_allKeys[-1].name  # just the last key pressed
                Resp3Space.rt = _Resp3Space_allKeys[-1].rt
                Resp3Space.duration = _Resp3Space_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=Finish,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            Finish.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if Finish.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in Finish.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Finish" ---
    for thisComponent in Finish.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for Finish
    Finish.tStop = globalClock.getTime(format='float')
    Finish.tStopRefresh = tThisFlipGlobal
    thisExp.addData('Finish.stopped', Finish.tStop)
    thisExp.nextEntry()
    # the Routine "Finish" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    # stop any playback components
    if thisExp.currentRoutine is not None:
        for comp in thisExp.currentRoutine.getPlaybackComponents():
            comp.stop()
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # return console logger level to WARNING
    logging.console.setLevel(logging.WARNING)
    # mark experiment handler as finished
    thisExp.status = FINISHED
    # run any 'at exit' functions
    for fcn in runAtExit:
        fcn()
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
