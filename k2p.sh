find "./output" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
python k2p.py ../../harbour/KNIME/Payment_prediction_2022 --out output/  --graph off
#python k2p.py ../../harbour/KNIME/ISU_Master --out output/ 
#python k2p.py tests/data/KNIME_traverse_order --out output/
#python k2p.py tests/data/KNIME_PP_2022_LR --out tests/data/!output/ --graph off