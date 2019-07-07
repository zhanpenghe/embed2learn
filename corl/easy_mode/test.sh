echo "$1"

for i in $(seq $2 5 $3)
do
    PYTHONPATH=/home/zhanpeng/Desktop/codes/embed2/embed2learn-private/embed2learn:/home/zhanpeng/Desktop/codes/embed2/embed2learn-private/external/metaworld pipenv run python /home/zhanpeng/Desktop/codes/embed2/embed2learn-private/corl/easy_mode/ten_tasks_test.py --pkl $1/itr_$i.pkl
done
