@ECHO OFF 
set CARLA_ROOT=C:\Users\alper\Desktop\CARLA
set SCENARIO_RUNNER_ROOT=C:\Users\alper\Desktop\CARLA\scenario_runner-0.9.11
set PYTHONPATH=%PYTHONPATH%;%CARLA_ROOT%\PythonAPI\carla\dist\carla-0.9.11-py3.7-win-amd64.egg
set PYTHONPATH=%PYTHONPATH%;%CARLA_ROOT%\PythonAPI\carla\agents
set PYTHONPATH=%PYTHONPATH%;%CARLA_ROOT%\PythonAPI\carla
set PYTHONPATH=%PYTHONPATH%;%CARLA_ROOT%\PythonAPI
:loop
python manual_control.py
pause
goto loop
