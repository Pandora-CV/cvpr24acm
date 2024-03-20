CUDA_VISIBLE_DEVICES=7 python evaluate.py --operation HRSC_MS_test \
					--heads 1 \
	                --model 50 \
					--coder acm \
              		--coder_cfg -1 \
              		--coder_mode model \
             		--box_loss riou \
			 		--weight_path checkpoint/hrsc/RIoU_Our_140.pth \
					--hrsc_test_size 640 \
	                --use_07_metric \
					--ap_thres 0.75
