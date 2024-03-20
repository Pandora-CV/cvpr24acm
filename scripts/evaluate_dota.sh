CUDA_VISIBLE_DEVICES=7 python evaluate.py --operation DOTA_test  \
	                --heads 15 \
	                --model 50 \
					--coder none \
              		--coder_cfg 1 \
              		--coder_mode loss \
             		--box_loss riou \
			 		--weight_path checkpoint/dota/KLD_140.pth \
					--test_image_dir /shared/datasets/DOTA/test4000/images \
                    --output_id 1
