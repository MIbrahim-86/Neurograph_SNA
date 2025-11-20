dataset="HCPGender"
batch_size="16"
model="rehub"
hidden="64"
main="main_new.py"
python $main --hidden $hidden--dataset $dataset --model $model --device 'cuda' --batch_size $batch_size --runs 1
