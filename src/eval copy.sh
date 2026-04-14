BASE_CMD="python main.py"
echo "开始运行 7 个任务..."

for task_id in {1..7}; do
    echo "启动 Task $task_id ..."
    $BASE_CMD --task_id $task_id
done
wait

echo "所有任务已完成！"