from run_eval import run_eval

output_dir = '/home/ubuntu/predictions'

run_eval(model='mistral-instruct', output_dir=output_dir, batch_size=10, json_mode=False)
run_eval(model='mistral-instruct', output_dir=output_dir, batch_size=1, json_mode=True)
run_eval(model='mistral-orca', output_dir=output_dir, batch_size=10, json_mode=False)
run_eval(model='mistral-orca', output_dir=output_dir, batch_size=1, json_mode=True)