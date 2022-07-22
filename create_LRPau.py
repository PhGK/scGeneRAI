import pandas as pd
import numpy as np
import sys
import os

print(sys.argv[1], sys.argv[2])
DATAPATH = sys.argv[1] + sys.argv[2]
OUTPUTPATH = sys.argv[3] + sys.argv[2] + '.parquet'

test_data = pd.read_csv(DATAPATH)
test_data['LRP'] = np.abs(test_data['LRP'])

sym_dir = test_data[['source_gene', 'target_gene', 'sample_name', 'LRP', 'y_pred', 'y', 'inpv']]
sym_trans = sym_dir.copy()
sym_trans.columns = ['target_gene', 'source_gene', 'sample_name', 'LRP', 'ty_pred', 'ty', 'tinpv']

sym = sym_dir.merge(sym_trans, on =['source_gene', 'target_gene', 'sample_name'])

sym['LRP'] = sym.apply(lambda row: np.minimum(row.LRP_x, row.LRP_y), axis=1)
#sym['LRP'] = sym.apply(lambda row: np.mean(row.LRP_x, row.LRP_y), axis=1)


output = sym[sym['source_gene']>sym['target_gene']]
output['sample_name'] = output['sample_name'].astype('category')
output['source_gene'] = output['source_gene'].astype('category')
output['target_gene'] = output['target_gene'].astype('category')
output['LRP'] = pd.to_numeric(output['LRP'], downcast='float')


result = output[['LRP', 'source_gene', 'target_gene', 'sample_name', 'inpv', 'tinpv']]

if not os.path.exists(sys.argv[3]):
        os.makedirs(sys.argv[3])
print('path:', sys.argv[3], OUTPUTPATH)
result.to_parquet(OUTPUTPATH)

