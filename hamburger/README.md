# Hamburger reconstruction files

#Prompt
python launch.py \
    --config configs/dreamfusion-sd.yaml \
    --train \
    --gpu 0 \
    system.prompt_processor.prompt="a delicious hamburger"
