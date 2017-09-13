python main.py --momentum 0 > worker_1_mom_0_l2_5en4.txt
python main.py --momentum 0.9 > worker_1_mom_0p9_l2_5en4.txt
python main.py --momentum 0 --decay 0 > worker_1_mom_0_l2_0.txt
python main.py --momentum 0.9 --decay 0 > worker_1_mom_0p9_l2_0.txt
# python main.py --batch-size 64 --epoch 10 --momentum 0 > worker_1_mom_0.txt
# python main.py --batch-size 64 --epoch 10 --momentum 0.9 > worker_1_mom_0p9.txt

# mkdir ma_result
# for n in {3..8}
# do
#   for i in {1..50}
#   do
#     mpiexec -n ${n} python main.py --batch-size $((n * 64)) --epoch 10 --momentum 0 --optim ma > ma_result/worker_${n}_mom_0_${i}.txt
#     mpiexec -n ${n} python main.py --batch-size $((n * 64)) --epoch 10 --momentum 0.9 --optim ma > ma_result/worker_${n}_mom_0p9_${i}.txt
#   done
# done

# mkdir dpsgd_result
# for n in {3..8}
# do
#   for i in {1..10}
#   do
#     mpiexec -n ${n} python main.py --batch-size $((n * 64)) --epoch 10 --momentum 0 --optim dpsgd > dpsgd_result/worker_${n}_mom_0_${i}.txt
#     # mpiexec -n ${n} python main.py --batch-size $((n * 64)) --epoch 10 --momentum 0.9 --optim dpsgd > dpsgd_result/worker_${n}_mom_0p9_${i}.txt
#   done
# done

# mkdir g_result
# for n in {3..8}
# do
#   for i in {1..50}
#   do
#     mpiexec -n ${n} python main.py --batch-size $((n * 64)) --epoch 10 --momentum 0 --optim super > g_result/worker_${n}_mom_0_${i}.txt
#     mpiexec -n ${n} python main.py --batch-size $((n * 64)) --epoch 10 --momentum 0.5 --optim super > g_result/worker_${n}_mom_0p5_${i}.txt
#     mpiexec -n ${n} python main.py --batch-size $((n * 64)) --epoch 10 --momentum 0.9 --optim super > g_result/worker_${n}_mom_0p9_${i}.txt
#   done
# done

# for n in {3..8}
# do
#   for i in {1..50}
#   do
#     mv dpsgd_result/worker_${n}_mom_0_${i}.txt dpsgd_result/worker_${n}_mom_0p9_${i}.txt
#     # mpiexec -n ${n} python main.py --batch-size $((n * 64)) --epoch 10 --momentum 0 --optim dpsgd > dpsgd_result/worker_${n}_mom_0_${i}.txt
#     # mpiexec -n ${n} python main.py --batch-size $((n * 64)) --epoch 10 --momentum 0.9 --optim dpsgd > dpsgd_result/worker_${n}_mom_0p9_${i}.txt
#   done
# done


# mkdir sgd_result
# for i in {1..50}
# do
#   python main.py --batch-size 64 --epoch 10 --momentum 0  > sgd_result/sgd_mom_0_${i}.txt
#   python main.py --batch-size 64 --epoch 10 --momentum 0.9  > sgd_result/sgd_mom_0p9_${i}.txt
# done

# mkdir dpsgd2_result
# for n in {4,6,8}
# do
#   for i in {1..50}
#   do
#     mpiexec -n ${n} python main.py --batch-size $((n * 64)) --epoch 10 --momentum 0 --optim dpsgd2 > dpsgd2_result/worker_${n}_mom_0_${i}.txt
#     mpiexec -n ${n} python main.py --batch-size $((n * 64)) --epoch 10 --momentum 0.9 --optim dpsgd2 > dpsgd2_result/worker_${n}_mom_0p9_${i}.txt
#   done
# done

#################################fashion#################################
# mkdir fashion_result/sgd_result
# for i in {1..5}
# do
#   python main.py --batch-size 64 --epoch 10 --momentum 0 --dataset fashion > fashion_result/sgd_result/worker_1_mom_0_${i}.txt
#   python main.py --batch-size 64 --epoch 10 --momentum 0.9 --dataset fashion > fashion_result/sgd_result/worker_1_mom_0p9_${i}.txt
# done

# mkdir fashion_result/ma_result
# for n in {4,6,8}
# do
#   for i in {1..5}
#   do
#     mpiexec -n ${n} python main.py --batch-size $((n * 64)) --epoch 10 --dataset fashion --momentum 0 --optim ma > fashion_result/ma_result/worker_${n}_mom_0_${i}.txt
#     mpiexec -n ${n} python main.py --batch-size $((n * 64)) --epoch 10 --dataset fashion --momentum 0.9 --optim ma > fashion_result/ma_result/worker_${n}_mom_0p9_${i}.txt
#   done
# done

# mkdir fashion_result/dpsgd_result
# for n in {4,6,8}
# do
#   for i in {1..5}
#   do
#     mpiexec -n ${n} python main.py --batch-size $((n * 64)) --epoch 10 --dataset fashion --momentum 0 --optim dpsgd > fashion_result/dpsgd_result/worker_${n}_mom_0_${i}.txt
#     mpiexec -n ${n} python main.py --batch-size $((n * 64)) --epoch 10 --dataset fashion --momentum 0.9 --optim dpsgd > fashion_result/dpsgd_result/worker_${n}_mom_0p9_${i}.txt
#   done
# done

# mkdir fashion_result/dpsgd2_result
# for n in {4,6,8}
# do
#   for i in {1..5}
#   do
#     mpiexec -n ${n} python main.py --batch-size $((n * 64)) --epoch 10 --dataset fashion --momentum 0 --optim dpsgd2 > fashion_result/dpsgd2_result/worker_${n}_mom_0_${i}.txt
#     mpiexec -n ${n} python main.py --batch-size $((n * 64)) --epoch 10 --dataset fashion --momentum 0.9 --optim dpsgd2 > fashion_result/dpsgd2_result/worker_${n}_mom_0p9_${i}.txt
#   done
# done
