# CIL
ETHZ CIL Monocular Depth Estimation 2025


The pipeline consists of four steps:
1. Fine-tune a base MiDaS model with uncertainty
2. Fine-tune expert MiDaS models with uncertainty
3. Predict depth maps and uncertainty maps for train/val/test images and store them in predictions_temp
4. Learn Metamodel by loading all predictions/uncertainties and utilizing Mixture of Experts, Uncertainty Masking, and Inverse Uncertainty Averaging


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

    expert_model_living_room_finetuned.pth
    expert_model_kitchen_finetuned.pth
    expert_model_dorm_room_finetuned.pth
    expert_model_home_office_finetuned.pth
    expert_model_bathroom_finetuned.pth

To obtain all predictions for the small training/validation list (you can add more images to them) and all test data (stored in src/predictions_temp), call.
We originally worked with --train-list train_list.txt --val-list val_list.txt (requiring almost 200 GB RAM for every training image), so we recommend instead using --train-list train_list_small.txt --val-list val_list_small.txt:

    python src/predict_base_and_expert_models.py --base-model-path models/base_model_with_uncertainty_finetuned_epoch_10.pth --train-list train_list_small.txt --val-list val_list_small.txt
    --predictions-temp-root /work/scratch/<user>/predictions_temp --cluster-root /cluster/courses/cil/monocular_depth/data

You need to specify your ethz username <user> on the cluster so that the data gets stored at scratch on the cluster (we have only 100 GB of space per person, so do not predict on more than 5k images for now or store otherwise locally)


4. Learn Metamodel by loading all predictions/uncertainties and utilizing Mixture of Experts, Uncertainty Masking, and Inverse Uncertainty Averaging

Then call the final metamodel training with:

    python src/finetune_metamodel_mask.py --train-list train_list_small.txt --val-list val_list_small.txt -b 8 --predictions-temp-root /work/scratch/<user>/predictions_temp --cluster-root /cluster/courses/cil/monocular_depth/data

The model is stored under models/final_metamodel.pth and can be loaded from there