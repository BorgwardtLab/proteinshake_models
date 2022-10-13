sbatch --cpus-per-task 2 -p gpu --gres=gpu:1 --mem-per-cpu 30G --output="$SCRIPTPATH/logs/$EMB.log" --wrap="python -m src.train --task PT --epochs 10 --representation voxel --batchsize 12"

sbatch --cpus-per-task 2 -p gpu --gres=gpu:1 --mem-per-cpu 30G --output="$SCRIPTPATH/logs/$EMB.log" --wrap="python -m src.train --task PT --epochs 10 --representation point --batchsize 32"
