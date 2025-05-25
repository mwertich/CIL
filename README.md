# CIL
ETHZ CIL Monocular Depth Estimation 2025


The pipeline consists of four steps, all performed on the cluster in the environment /cluster/courses/cil/monocular_depth:
1. Fine-tune a base MiDaS model with uncertainty
2. Fine-tune expert MiDaS models with uncertainty
3. Predict depth maps and uncertainty maps for train/val/test images and store them in predictions_temp
4. Learn Metamodel by loading all predictions/uncertainties and utilizing Mixture of Experts Ensembling, and Inverse Uncertainty Averaging


1. Fine-tune a base MiDaS model with uncertainty
    python src/finetune_model_with_uncertainty.py

The base model is stored as "models/model_{run_id}_finetuned.pth". Rename it to "base_model_with_uncertainty_finetuned.pth"

2. Fine-tune expert MiDas models with uncertainty:

Call src/categorize.py to obtain the categorized list of images belonging to the respective category.

There exist five categories from which we have obtained images:  ["sleeping", "work", "kitchen", "living", "remaining"]

To train an expert model with uncertainty: 

    python src/finetune_model_with_uncertainty.py -trainl category_lists/{category}_train_list.txt -vall category_lists/{category}_val_list.txt -f

The expert model is stored as "models/model_{run_id}_finetuned.pth". Rename it to "model_{cateogry}_finetuned.pth"

3. Predict depth maps and uncertainty maps for train/val/test images and store them in predictions_temp

Load and rename the models and store them in models/*

The base model is:

    base_model_with_uncertainty_finetuned.pth

The five expert models are:

    model_living_room_finetuned.pth
    model_living_finetuned.pth
    model_sleeping_finetuned.pth
    model_kitchen_finetuned.pth
    model_remaining_finetuned.pth

To obtain all predictions for the small training/validation list for all test data (which get stored in src/predictions_temp), simply call:

    python src/predict_base_and_expert_models.py --base-model-path models/base_model_with_uncertainty_finetuned.pth --train-list train_list_small.txt --val-list val_list_small.txt
    --predictions-temp-root /work/scratch/<user>/predictions_temp --cluster-root /cluster/courses/cil/monocular_depth/data

(You can also call the entire training and validation lists: --train-list train_list.txt --val-list val_list.txt (requiring almost 200 GB RAM to store all intermediate predictions), so we recommend using the smaller lists instead)

Furthermore, you need to specify your ethz username on the cluster so that the data gets stored at scratch on the cluster or point to a local path where predictions_temp can be stored:


4. Learn Metamodel by loading all predictions/uncertainties and utilizing Mixture of Experts Ensembling, and Inverse Uncertainty Averaging

Then call the final metamodel training with:

    python src/finetune_metamodel.py --train-list train_list_small.txt --val-list val_list_small.txt --predictions-temp-root /work/scratch/<user>/predictions_temp --cluster-root /cluster/courses/cil/monocular_depth/data

The model is stored under models/final_metamodel.pth and can be loaded from there