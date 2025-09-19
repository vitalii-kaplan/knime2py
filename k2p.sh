find "./output" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
#python k2p.py ../../harbour/KNIME/Payment_prediction_2022 --out output/  --graph off
#python k2p.py ../../harbour/KNIME/HW-Churn --out output/ --workbook py --graph off
#python k2p.py ../../harbour/KNIME/ISU_Master --out output/ 
#python k2p.py tests/data/KNIME_traverse_order --out output/
#python k2p.py tests/data/KNIME_PP_2022_LR --out tests/data/!output/ --graph off
python k2p.py ../../harbour/KNIME/Churn_prediction_LR --out output/ --workbook py --graph off