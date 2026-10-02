@echo off
REM Repeatable policy eval: for each episode -> place ball -> go home -> record 1 episode.
REM Run from the VLA folder inside the (lerobot) conda prompt:  run_eval.bat
REM Edit the settings below for each eval run.

setlocal EnableDelayedExpansion

set EVAL_NAME=eval_v7
set POLICY=outputs/train/act_so101_ball_v7/checkpoints/last/pretrained_model
set N=10


REM Snap the policy's gripper output: below 60 -> 45 (closed), else 75 (open).
REM Use for models trained on so101_ball_v4 (binary gripper labels). Delete this line for older models.
set GRIPPER_SNAP=60,45,75

set CAMERAS="{ fixed:{type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}, handeye:{type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30} }"
set DATASET_DIR=%USERPROFILE%\.cache\huggingface\lerobot\local\%EVAL_NAME%

for /L %%i in (1,1,%N%) do (
    echo.
    echo ===== Episode %%i of %N% =====
    echo Place the ball and check the green circle, then press any key. Ctrl+C to quit.
    pause >nul

    python go_home.py
    if errorlevel 1 (
        echo go_home failed - fix the arm, then press any key to retry this run.
        pause >nul
    )

    if exist "!DATASET_DIR!" (set RESUME=--resume=true) else (set RESUME=)

    python -m lerobot.record --robot.type=so101_follower --robot.port=COM24 --robot.id=my_awesome_follower_arm --robot.cameras=%CAMERAS% --policy.path=%POLICY% --policy.device=cuda --policy.temporal_ensemble_coeff=0.01 --policy.n_action_steps=1 --dataset.repo_id=local/%EVAL_NAME% --dataset.push_to_hub=false --dataset.num_episodes=1 --dataset.single_task="Pick up the red ball and place it on the green circle" --display_data=true !RESUME!
)

echo.
echo All %N% episodes done. Saved to %DATASET_DIR%
endlocal
