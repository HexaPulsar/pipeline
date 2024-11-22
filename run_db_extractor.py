from data_preprocessing.DBExtractor import DBExtractor
from data_preprocessing.CreateAstroObjectPKL import CreateAstroObjectPKL
#from ao2atat import AO2ATAT
#
#DBExtractor(path_oids_to_pull='/home/mdelafuente/SSL/2020_oids.parquet',
#            path_to_save_dir='/home/mdelafuente/SSL/2020/pulled_data/')


CreateAstroObjectPKL(chunks_dir='/home/mdelafuente/SSL/pulled_data/',
                     output_dir='/home/mdelafuente/SSL/aos/')