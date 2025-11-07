virtualEnv = '/home/philipp/projects/stamp_forming_sim/paraview_venv/bin/activate_this.py'
exec(open(virtualEnv).read(), {'__file__': virtualEnv})
import subprocess

# Execute pip freeze and capture the output
result = subprocess.run(["pip", "freeze"], capture_output=True, text=True)

# Print the output
print(result.stdout)