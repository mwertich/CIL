# CIL
ETHZ CIL Monocular Depth Estimation 2025


The pipeline consists of four steps:
1. Fine-tune a base MiDaS model with uncertainty (potentially pre-trained a little bit without uncertainty)
2. Fine-tune expert MiDaS models with uncertainty
3. Predict depth maps and uncertainty maps for train/val/test images and store them in predictions_temp
4. Learn Metamodel by loading all predictions/uncertainties and utilizing Mixture of Experts, Uncertainty Masking, and Inverse Uncertainty Sampling


1. Fine-tune a base MiDaS model with uncertainty
    python src/finetune_model_with_uncertainty.py

The base model is base_model_with_uncertainty_finetuned_epoch_10.pth

2. Fine-tune expert MiDas models with uncertainty:

Call src/categorize.py to obtain the categorized list of images belonging to the respective category.

There exist five categories from which we have obtained images: kitchen, bathroom, dorm_room, home_office, and living_room

To train an expert model with uncertainty: 

    python src/finetune_model_with_uncertainty.py --trainl category_lists/{category}_train_list.txt --vall category_lists/{category}_val_list.txt

3. Predict depth maps and uncertainty maps for train/val/test images and store them in predictions_temp

Load the models and store them in models/*

The base model is:

    base_model_with_uncertainty_finetuned_epoch_10.pth

The five expert models are:

    model_living_room_finetuned.pth
    model_kitchen_finetuned.pth
    model_dorm_room_finetuned.pth
    model_home_office_finetuned.pth
    model_bathroom_finetuned.pth

To obtain all predictions for the small training/validation list (you can add more images to them) and all test data (stored in src/predictions_temp), call:

    python src/predict_base_and_expert_models.py --base-model-path models/base_model_with_uncertainty_finetuned_epoch_10.pth --train-list train_list_small.txt --val-list val_list_small.txt
    --predictions-temp-root /work/scratch/<user>/predictions_temp

You need to specify your ethz username <user> on the cluster so that the data gets stored at scratch on the cluster (we have only 100 GB of space per person, so do not predict on more than 5k images for now or store otherwise locally)


4. Learn Metamodel by loading all predictions/uncertainties and utilizing Mixture of Experts, Uncertainty Masking, and Inverse Uncertainty Sampling

Then call the final metamodel training with:

    python src/predict_base_and_expert_models.py --train-list train_list_small.txt --val-list val_list_small.txt

Additional interesting args are:
    parser.add_argument("--uncertainty-threshold", type=float, default=0.05, help="Only evaluate loss at uncertain regions (uncertainty > threshold), otherwise base model")
    parser.add_argument("--alpha", type=float, default=1., help="mse loss of masked region")
    parser.add_argument("--beta", type=float, default=1., help="cross entropy loss of masked region (with best expert per pixel)")
    parser.add_argument("--gamma", type=float, default=1., help="entropy regularization (punishes flat distributions in terms of post-softmax values)")
    parser.add_argument("--tau", type=float, default=4., help="temperature of model outputs before softmax (logits)")

The model is stored under models/final_metamodel.pth and can be loaded from there