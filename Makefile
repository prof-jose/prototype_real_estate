test_reproduce:
	python train.py --config data/king_county_per_sqft.json --n_prototypes 100 --epochs 5  --learning-rate 0.01 --scale 0.1 --regularization 0 --logdir tmp --batch-size 256 --seed 12 | grep RESULTS | sed -e "s@^.*[RESULTS\]@@g" > aux1.txt && python train.py --config data/king_county_per_sqft.json --n_prototypes 100 --epochs 5  --learning-rate 0.01 --scale 0.1 --regularization 0 --logdir tmp --batch-size 256 --seed 12 | grep RESULTS | sed -e "s@^.*[RESULTS\]@@g" > aux2.txt && diff aux1.txt aux2.txt && rm aux1.txt aux2.txt && echo "Test reproduce = OK"

## TARGETS RELATED TO DATA PREPARATION

datasets_per_sqft: 
	python data/scripts/price_per_sqft.py


## TARGETS RELATED TO EXPERIMENTS
PROTOTYPES=50
CONFIG=data/king_county_per_sqft.json
PREFIX=
EPOCHS=10
LR=0.01
SCALE=0.01
BATCH=256
SEED=0

LR_SEQ = 1e-1 5e-1 1e-2 5e-3 1e-3 5e-4
SCALE_SEQ = 1 0.5 1e-1 5e-2 1e-2 5e-3 1e-3 5e-4
BATCH_SEQ = 32 64 128 256 512

experiment: 
	mkdir -p logs
	python train.py --config $(CONFIG) --n_prototypes $(PROTOTYPES) --epochs $(EPOCHS) --learning-rate $(LR) --scale $(SCALE) --regularization 0.0 --batch-size $(BATCH) --seed $(SEED) > logs/$(PREFIX)run_$(EPOCHS)_$(LR)_$(SCALE)_$(BATCH).log

experiment_kmeans: 
	mkdir -p logs
	python train.py --config $(CONFIG) --n_prototypes $(PROTOTYPES) --epochs 0 --seed $(SEED) > logs/$(PREFIX)run_kmeans.log

experiment_random:
	mkdir -p logs
	python train.py --config $(CONFIG) --n_prototypes $(PROTOTYPES) --epochs 0 --seed $(SEED) --init-method random_pick > logs/$(PREFIX)run_random.log

validation:
	for lr in $(LR_SEQ); do \
		for scale in $(SCALE_SEQ); do \
			for batch in $(BATCH_SEQ); do \
				make experiment LR=$${lr} SCALE=$${scale} BATCH=$${batch}; \
			done; \
		done; \
	done 

# Do the 10 runs of the validation experiment
validation_n:
	for seed in `seq 0 9`; do \
		mkdir -p logs/n_$(PROTOTYPES)/seed_$${seed}; \
		make validation SEED=$${seed} PREFIX=n_$(PROTOTYPES)/seed_$${seed}/; \
	done

validation_kmeans:
	for seed in `seq 0 9`; do \
		mkdir -p logs/k_means/seed_$${seed}; \
		make experiment_kmeans SEED=$${seed} PREFIX=k_means/seed_$${seed}/; \
	done

validation_random:
	for seed in `seq 0 9`; do \
		mkdir -p logs/random/seed_$${seed}; \
		make experiment_random SEED=$${seed} PREFIX=random/seed_$${seed}/; \
	done


clean: 
	rm -rf mlruns
