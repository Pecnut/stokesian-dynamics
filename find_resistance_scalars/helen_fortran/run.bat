ECHO OFF
CLS
TITLE Helen's 2-sphere code
ECHO Helen's 2-sphere code is about to compile and execute.
ECHO.
ECHO Compiling (please wait)...
g95 2sphere.f base.f reflect.f -o output.exe 
ECHO Compilation complete.
ECHO.
ECHO Executing (please wait)...
REM output.exe takes two arguments: (1) the setup to run, (2) to output in "words" or in "mathematica"
output.exe 3.9 1 1 F11 words > output.txt
ECHO Execution complete.
ECHO.
ECHO Output can be found in output.txt.
ECHO.
PAUSE