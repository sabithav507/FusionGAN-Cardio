from pathlib import Path
import pandas as pd
import pytest
import numpy as np
from PIL import Image
from fusiongan_cardio.data import load_manifest,prepare_tmed,preprocess_image,audit_splits

# Balancing is deliberately defined in quality.py, not data.py.
from fusiongan_cardio.quality import balancing_deficits

def write_png(path):Image.fromarray(np.ones((10,15),dtype=np.uint8)*80).save(path)

def test_manifest_leakage_and_synthetic_only_train(tmp_path):
    img=tmp_path/'x.png';other=tmp_path/'y.png';write_png(img);write_png(other)
    df=pd.DataFrame([{'path':str(img),'split':'train','subject_id':'patient1','study_id':'A','y':'no_as','view':'A2C','is_synthetic':False},
                     {'path':str(other),'split':'test','subject_id':'patient1','study_id':'B','y':'no_as','view':'A2C','is_synthetic':False}])
    f=tmp_path/'manifest.csv';df.to_csv(f,index=False)
    with pytest.raises(ValueError,match='LEAKAGE'):load_manifest(f)
    df.loc[1,'subject_id']='patient2';df.loc[1,'is_synthetic']=True
    df.to_csv(f,index=False)
    with pytest.raises(ValueError,match='LEAKAGE'):load_manifest(f)


def test_patient_grouping_and_deficit(tmp_path):
    rows=[]
    for j in range(8):
        image=tmp_path/f'{j}.png';write_png(image)
        rows.append({'image_path':image.name,'study_id':f's{j}','patient_id':f'p{j//2}',
                     'disease':'no_as' if j<6 else 'early_as','view':'A4C'})
    meta=tmp_path/'meta.csv';pd.DataFrame(rows).to_csv(meta,index=False)
    out=tmp_path/'out.csv'
    prepare_tmed(meta,out,seed=42,max_frames_per_view_study=2)
    df=load_manifest(out)
    assert audit_splits(df)['subjects']==4
    counts,target,deficits=balancing_deficits(df,ratio=.9)
    assert target>=1
    assert all(d>=0 for d in deficits.values())


def test_preprocess_letterbox_no_reflection():
    arr=np.full((7,13),125,dtype=np.uint8);arr[:,0]=255
    out=np.asarray(preprocess_image(Image.fromarray(arr),32))
    assert out.shape==(32,32)
    assert out[0,:].sum()==0
    assert out[31,:].sum()==0
    assert out[16,0]>out[16,31]
