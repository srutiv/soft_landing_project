from omxdsm import write_xdsm

# Assuming you have an OpenMDAO-Dymos problem defined as 'p' or 'prob'
# Replace this with your actual OpenMDAO-Dymos problem or model
# ...

# Call the write_xdsm function to generate the XDSM diagram
write_xdsm(prob, "my_system_xdsm")  # Provide your OpenMDAO problem/model and a filename
