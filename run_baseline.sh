dataset="HCPGender"
batch_size="16"
epochs="100"
mid_dim="48"
main="main.py"
python $main --dataset $dataset --epochs $epochs --batch_size $batch_size --mid_dim $mid_dim
