from pathlib import Path

class Config:
    
    # Experiment and Path Configuration
    log_base_dir = "experiments"
    experiment_name = "beit3_ftcoco"  # Name of the experiment

    dialogue_format = "VisDial"   # 'Summarized' or 'VisDial'
    dialogue_round = 10
    use_random_rounds = True
    use_caption_masking = False
    caption_masking_prob = 0.2

    train_json_path = './data/visdial_1.0_train.json' 
    val_corpus_json_path = './data/ChatIR_Protocol/Search_Space_val_50k.json'
    val_queries_path = './data/dialogues/VisDial_v1_0_queries_val.json'
    val_generated_image_dir = './data/generated_images/VisDial_v1_0_queries_val/your_generated_images'    
    train_reference_image_dir = './data/query_images'

    beit3_checkpoint_path = "./model/beit3_base_itc_patch16_224.pth"
    beit3_tokenizer_path = "./model/beit3.spm"

    num_epochs = 50
    batch_size = 128
    update_freq = 8   
    val_batch_size = 32
    beit3_lr = 1e-5
    warmup_epochs = 5    
    validation_frequency = 1
    weight_decay = 0.05   
    layer_decay = 0.90  
    drop_path = 0.2     
    clip_grad = 3.0     
    model_ema = False   
    model_ema_decay = 0.999 
    
    resume_from =  None 
    save_training = True 

    input_size = 224  
    train_interpolation = 'bicubic'  
    randaug = False  
    loss_components = ["ref_tgt", "text_tgt", "fused_tgt", "ref_text","dist_agreement"]
    loss_weights = [1.0,1.0,1.0,0.5,0.2] 

    use_learnable_weights = True  # Whether to convert loss_weights into learnable parameters
    dist_loss_temp = 1.0           

    use_hnm = False          
    hnm_weight = 0.1         # lambda_h: Weight of HNM loss in total loss
    hnm_topk = 4             # K: Select top-K hardest negatives
    hnm_margin = 0.1         # m: Margin between positive and negative samples
    hnm_temp = 0.1              # tau_h: Temperature coefficient for HNM loss

    wandb_entity = None  # Username or Team name
    wandb_project = "your_project_name" # Project name
    wandb_mode = "online" # 'online', 'offline', 'disabled'

    

