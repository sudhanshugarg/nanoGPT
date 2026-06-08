# train a miniature character-level copy task model
# good for debugging and playing on macbooks and such

out_dir = 'out-copy'
eval_interval = 500 # keep frequent because we'll overfit
eval_iters = 200
log_interval = 10 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = True

wandb_log = False # override via command line if you like
wandb_project = 'copy-task'
wandb_run_name = 'copy-sugarg'

dataset = 'copy'
gradient_accumulation_steps = 1
batch_size = 2
block_size = 12 # context of up to 10 previous characters

# baby GPT model :)
n_layer = 2
n_head = 1
n_embd = 384
dropout = 0.2

learning_rate = 1e-3 # with baby networks can afford to go a bit higher
max_iters = 5000
lr_decay_iters = 5000 # make equal to max_iters usually
min_lr = 1e-4 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 100 # not super necessary potentially

# on macbook also add
device = 'cpu'  # run on cpu only
compile = False # do not torch compile the model
max_new_tokens = 40
temperature = 0.8
top_k = 6
init_from = 'scratch'
