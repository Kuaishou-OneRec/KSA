import subprocess


lines = subprocess.run(
    ["ps -ef | grep -v '<defunct>' | grep 'recipes/' | grep -v 'grep' | awk '{print $2}' | xargs kill -9"],
    capture_output=True, shell=True, text=True
)
