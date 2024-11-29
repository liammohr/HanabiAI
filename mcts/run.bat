@echo off

REM Define the output file and Python script
set output_file=outputs.txt
set python_script=mcts_main.py

REM Ensure the output file is empty initially
echo. > %output_file%

REM Loop to run the Python script 100 times
for /L %%i in (1,1,100) do (
    REM Execute the Python script and capture its output
    for /f "delims=" %%j in ('python %python_script%') do (
        echo %%j >> %output_file%
    )
)

echo Execution completed. Results saved in %output_file%.
