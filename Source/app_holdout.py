"""TriFuse-AD: frozen 15-checkpoint hold-out ensemble."""
import json
import time
import hashlib
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from core import holdout as E, preproc as PP, report as R, viz, history as H

st.set_page_config(page_title='TriFuse-AD',page_icon='🧠',layout='wide')
CLASSES=E.CLASSES

# Native headings remain semantic; the accent works with both light and dark themes.
st.markdown('''<style>
[data-testid="stHeadingWithActionElements"] h1 {
    border-left: 5px solid #33b7ab;
    padding-left: 0.8rem;
    letter-spacing: -0.035em;
    line-height: 1.25;
}
[data-testid="stHeadingWithActionElements"] h3 {
    border-left: 3px solid #33b7ab;
    padding-left: 0.7rem;
    margin-top: 1rem;
    line-height: 1.45;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    font-weight: 650;
}
</style>''',unsafe_allow_html=True)
CLASS_LABELS={'CN':'Nhận thức bình thường','VMD':'Sa sút trí tuệ rất nhẹ','AD':'Nhóm Alzheimer'}
FEATURE_HELP={
    'age':'Tuổi tại thời điểm chụp MRI, tính bằng năm.',
    'sex':'Mã hóa trong bộ dữ liệu: 0 = nữ, 1 = nam.',
    'education':'Mức học vấn OASIS: 1–5; không nhập số năm đi học.',
    'ses':'Điều kiện kinh tế xã hội: 1 = cao nhất, 5 = thấp nhất. Chọn thiếu nếu chưa biết.',
    'etiv':'Thể tích nội sọ ước tính, đơn vị mm³; lấy từ kết quả xử lý MRI.',
    'nwbv':'Thể tích não chuẩn hóa, dạng tỷ lệ (ví dụ 0.74), không nhập 74.',
    'asf':'Hệ số chuẩn hóa atlas từ xử lý MRI (ví dụ 1.21).',
}

@st.cache_resource(show_spinner='Đang xác minh và nạp 15 checkpoint…')
def get_engine(device='auto'):
    return E.get_default_engine() if device=='auto' else E.HoldoutEnsemble(device=device)

@st.cache_data(show_spinner=False)
def get_cases():
    return pd.read_csv(E.OUTPUT/'locked_test_manifest.csv').set_index('subject_id',drop=False)

@st.cache_data(show_spinner=False)
def read_result(name):
    return json.loads((E.OUTPUT/name).read_text(encoding='utf-8'))

def input_signature(source, sid, uploaded, clinical, patient_id):
    payload={f:None if pd.isna(clinical[f]) else clinical[f] for f in E.FEATURES}
    image=(sid or '') if uploaded is None else hashlib.sha256(uploaded.getvalue()).hexdigest()
    return hashlib.sha256(json.dumps([source,image,payload,patient_id],sort_keys=True).encode()).hexdigest()

def preview_slices(raw):
    fig,axes=plt.subplots(1,3,figsize=(10,3))
    for ax,i,label in zip(axes,[4,13,22],['Axial','Coronal','Sagittal']):
        ax.imshow(raw[i],cmap='gray');ax.set_title(label);ax.axis('off')
    fig.tight_layout();st.pyplot(fig);plt.close(fig)

st.title('TriFuse-AD')
st.caption(E.MODEL_NAME)
st.caption('Tải MRI → Nhập thông tin → Xem dự đoán → Tải báo cáo')
st.sidebar.warning('Công cụ nghiên cứu; kết quả không phải chẩn đoán y khoa.')
cases=get_cases()
st.sidebar.write('15 checkpoint · 5 fold × 3 seed')
st.sidebar.caption('Ảnh và dữ liệu bảng dùng tiền xử lý đóng băng theo từng fold.')
st.sidebar.caption('CN: nhận thức bình thường · VMD: sa sút trí tuệ rất nhẹ · AD: nhóm Alzheimer')
st.sidebar.caption('Xử lý 15 mô hình có thể mất thời gian. Lần đầu cần xác minh trọng số.')
with st.sidebar.expander('Thông tin hệ thống'):
    st.write('Mô hình: Full TriFuse-AD, trung bình 15 checkpoint')
    st.write('Seed: 42, 123, 2024 · Fold: 0–4')
    import torch
    st.write('Thiết bị: '+(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'))
    processing_device=st.selectbox('Chế độ xử lý',['Tự động (ưu tiên GPU)','CPU'],help='Chọn CPU nếu GPU báo lỗi hoặc đang dùng cho huấn luyện.')
tabs=st.tabs(['Giới thiệu','Dự đoán','Kết quả thực nghiệm','Grad-CAM','Xuất báo cáo','Lịch sử ca'])

with tabs[0]:
    st.subheader('Phạm vi')
    st.write('Phân loại ba trạng thái CN, VMD và AD từ 27 lát cắt MRI cùng bảy đặc trưng dạng bảng. Dự đoán là trung bình đều xác suất của 15 checkpoint hold-out; không chọn mô hình theo ca tải lên.')
    st.write('Tập phát triển gồm 159 người OASIS-1; 39 người được giữ riêng khỏi lần huấn luyện mới. Hold-out được tách hồi cứu từ dữ liệu từng dùng trước đây. Kết quả ADNI mang tính thăm dò.')
    st.info('Xác suất phân loại tại thời điểm có dữ liệu chưa phải xác suất mắc Alzheimer trong tương lai. Grad-CAM là minh họa, chưa xác nhận vùng tổn thương.')
    st.caption('Grad-CAM hiển thị từ checkpoint seed 42, fold 0, với lớp mục tiêu của ensemble; bản đồ này không phải lời giải thích đã xác thực cho toàn ensemble.')
    st.subheader('Bắt đầu sử dụng')
    st.write('Mở tab Dự đoán, chọn một ca mẫu để làm quen hoặc tải MRI. Kiểm tra ảnh xem trước, bổ sung thông tin rồi bấm Thực hiện dự đoán. Kết quả có thể tải xuống tại tab Xuất báo cáo.')

with tabs[1]:
    st.subheader('1. Chọn ảnh MRI')
    source=st.radio('Nguồn ảnh MRI',['Tải lên tệp MRI từ máy tính','Chọn nhanh từ ngân hàng ca mẫu (OASIS-1 Demo)'],key='pred_mri_source')
    uploaded=None;sid=None
    if source.startswith('Tải lên'):
        uploaded=st.file_uploader('Tệp MRI: NIfTI hoặc NumPy',type=['nii','nii.gz','npy','npz'],key='pred_file_uploader')
        st.caption('NumPy 27 × 224 × 224 phải giữ thứ tự 9 axial, 9 coronal, 9 sagittal. Ảnh 2D đơn không dùng cho suy luận chính thức. Ảnh thể tích cần được kiểm tra hướng và chất lượng trước khi diễn giải.')
        with st.expander('Phân đoạn MRI 3D và căn chỉnh atlas MNI152'):
            st.write('Dùng ảnh T1 NIfTI 3D toàn đầu. FSL sẽ tách não, phân đoạn mô và căn chỉnh với mẫu MNI152. Bước này có thể mất vài phút.')
            st.caption('Trên Windows cần Ubuntu WSL và FSL. Kết quả xử lý được hiển thị riêng để kiểm tra. Các trường eTIV, nWBV và ASF vẫn lấy từ số đo tương thích OASIS-1 hoặc chọn “Chưa có”.')
            is_t1 = uploaded is not None and uploaded.name.lower().endswith(('.nii', '.nii.gz'))
            if is_t1:
                image_key = hashlib.sha256(uploaded.getbuffer()).hexdigest()
                processed = st.session_state.get('mri3d_result')
                if processed and processed[0] != image_key:
                    st.session_state.pop('mri3d_result')
                if st.button('Xử lý MRI 3D', key='process_mri3d'):
                    st.session_state.pop('mri3d_result', None)
                    try:
                        from core.mri3d import process_t1
                        with st.spinner('Đang chạy FSL trên ảnh 3D…'):
                            previews, archive = process_t1(uploaded.getvalue(), uploaded.name)
                        st.session_state['mri3d_result'] = (image_key, previews, archive)
                    except (OSError, ValueError, RuntimeError, ImportError) as exc:
                        st.error(f'Chưa xử lý được MRI 3D: {exc}')
                processed = st.session_state.get('mri3d_result')
                if processed:
                    _, previews, archive = processed
                    plane = st.selectbox('Mặt phẳng kiểm tra', ['Axial', 'Coronal', 'Sagittal'], key='mri3d_plane')
                    axis = ['Axial', 'Coronal', 'Sagittal'].index(plane)
                    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
                    for ax, key, title in zip(axes, ('atlas', 'brain', 'tissues'),
                                              ('Mẫu MNI152', 'MRI đã căn chỉnh', 'Phân đoạn mô')):
                        ax.imshow(previews[key][axis], cmap='gray' if key != 'tissues' else 'viridis',
                                  vmin=0, vmax=3 if key == 'tissues' else None,
                                  interpolation='nearest')
                        ax.set_title(title)
                        ax.axis('off')
                    fig.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                    st.caption('Nhãn phân đoạn: 1 = dịch não tủy, 2 = chất xám, 3 = chất trắng. Kiểm tra độ khớp với mẫu và ranh giới mô trước khi sử dụng.')
                    st.download_button('Tải ảnh phân đoạn và ma trận căn chỉnh', archive,
                                       file_name='mri3d_mni152.zip', mime='application/zip')
            else:
                st.caption('Tải file .nii hoặc .nii.gz để bật bước xử lý 3D.')
    else:
        sid=st.selectbox('Ca test OASIS đã lưu',cases.index.tolist(),key='pred_sample_subject')
        st.caption('Đây là các ca thuộc tập test đã đánh giá, dùng để minh họa; không phải dữ liệu mới.')
    patient_id=st.text_input('Mã ca',value='BN-TEST-001',key='pred_patient_id')
    st.caption('Nên dùng mã ca thay cho họ tên. Kết quả được lưu vào lịch sử trên máy này.')
    st.subheader('2. Thông tin lâm sàng')
    st.caption('Nhập thông tin thực tế. Nếu chưa có chỉ số, chọn “Chưa có”; giá trị mặc định chỉ phục vụ minh họa.')
    use_case=st.checkbox('Dùng thông tin gốc của ca mẫu',value=True,key='use_case_features') if sid else False
    clinical={}
    cols=st.columns(3)
    defaults={'age':75.,'sex':0.,'education':3.,'ses':2.5,'etiv':1450.,'nwbv':.74,'asf':1.21}
    for i,f in enumerate(E.FEATURES):
        if use_case:
            v=cases.loc[sid,f]
            clinical[f]=float(v) if pd.notna(v) else np.nan
            cols[i%3].write(f'{PP.FEATURE_LABELS_VI[f]}: '+('Thiếu (khôi phục theo từng fold)' if pd.isna(v) else str(v)))
        else:
            missing=cols[i%3].checkbox('Chưa có '+PP.FEATURE_LABELS_VI[f],key='missing_'+f)
            if f in ['sex','education','ses']:
                choices=[0.,1.] if f=='sex' else [1.,2.,3.,4.,5.]
                v=cols[i%3].selectbox(PP.FEATURE_LABELS_VI[f],choices,index=choices.index(defaults[f]) if defaults[f] in choices else 1,
                    format_func=(lambda x:'Nữ' if x==0 else 'Nam') if f=='sex' else str,
                    help=FEATURE_HELP[f],key='value_'+f,disabled=missing)
            else:
                v=cols[i%3].number_input(PP.FEATURE_LABELS_VI[f],min_value=0.,value=defaults[f],help=FEATURE_HELP[f],key='value_'+f,disabled=missing)
            clinical[f]=np.nan if missing else float(v)
    signature=input_signature(source,sid,uploaded,clinical,patient_id)
    previous=st.session_state.get('pred_result')
    if previous and previous.get('input_signature')!=signature:
        st.session_state.pop('pred_result',None)
        st.session_state.pop('exported_report',None)
        st.info('Đầu vào đã thay đổi. Hãy dự đoán lại để cập nhật kết quả và báo cáo.')
    issues=[]
    if not patient_id.strip():issues.append('Vui lòng nhập mã ca.')
    if not pd.isna(clinical['age']) and not 0<clinical['age']<=120:issues.append('Tuổi phải lớn hơn 0 và không quá 120.')
    if not pd.isna(clinical['nwbv']) and not 0<clinical['nwbv']<=1:issues.append('nWBV phải nằm trong (0, 1]; ví dụ 0.74.')
    for feature in ['etiv','asf']:
        if not pd.isna(clinical[feature]) and clinical[feature]<=0:issues.append(feature+' phải lớn hơn 0 hoặc chọn Chưa có.')
    for issue in issues:st.warning(issue)
    if sid is not None or uploaded is not None:
        if st.button('Kiểm tra và xem trước MRI',key='preview_mri'):
            try:
                with st.spinner('Đang đọc ảnh…'):
                    raw_preview=np.load(E.DATA/'triplanar_2d'/f'{sid}.npy',allow_pickle=False) if sid else PP.process_uploaded_mri(uploaded.getvalue(),uploaded.name)[1]
                    E.normalize_slices(raw_preview)
                    st.session_state['mri_preview']=(signature,raw_preview)
            except Exception as exc:st.error('Không đọc được MRI: '+str(exc))
    preview=st.session_state.get('mri_preview')
    if preview and preview[0]==signature:
        st.caption('Ba lát đại diện từ 27 lát đầu vào; đây là ảnh kiểm tra, chưa phải kết quả dự đoán.')
        preview_slices(preview[1])
    with st.expander('Tùy chọn giải thích bằng Grad-CAM'):
        with_cam=st.checkbox('Tạo bản đồ vùng ảnh hưởng đến dự đoán',value=False,key='pred_calc_cam')
        st.caption('Bật tùy chọn này sẽ tăng thời gian xử lý. Bản đồ lấy từ seed 42, fold 0; chưa xác nhận vị trí hồi hải mã hoặc tổn thương.')
    if st.button('Thực hiện dự đoán',type='primary',key='btn_run_pred',disabled=bool(issues) or (sid is None and uploaded is None)):
        if sid is None and uploaded is None:
            st.warning('Vui lòng tải lên tệp MRI hoặc chọn ca mẫu.')
        else:
            try:
                started=time.perf_counter()
                st.session_state.pop('pred_result',None)
                st.session_state.pop('exported_report',None)
                if sid:
                    raw=np.load(E.DATA/'triplanar_2d'/f'{sid}.npy',allow_pickle=False)
                    filename=sid+'.npy'
                else:
                    _,raw,info=PP.process_uploaded_mri(uploaded.getvalue(),uploaded.name)
                    filename=uploaded.name
                with st.spinner('Đang tổng hợp dự đoán từ 15 checkpoint…'):
                    engine=get_engine('cpu' if processing_device=='CPU' else 'auto')
                    bar=st.progress(0,text='Đã nạp checkpoint. Bắt đầu phân tích…')
                    completed=[0]
                    def update_progress(member):
                        completed[0]+=1
                        bar.progress(min(completed[0]/15,1.),text=f'Đã xử lý {completed[0]}/15 checkpoint')
                    try:
                        result=engine.predict(raw,clinical,with_cam=with_cam,progress=update_progress)
                    except RuntimeError as gpu_error:
                        gpu_failure=any(word in str(gpu_error).upper() for word in ['CUDNN','CUDA','CUBLAS'])
                        if engine.device.type!='cuda' or not gpu_failure:
                            raise
                        st.warning('GPU gặp lỗi. Đang chạy lại đủ 15 checkpoint bằng CPU; thời gian xử lý sẽ dài hơn.')
                        # Discard the CUDA module and rebuild from the verified CPU checkpoints.
                        engine.model=None
                        import gc
                        gc.collect()
                        try:torch.cuda.empty_cache()
                        except RuntimeError:pass
                        engine=get_engine('cpu')
                        completed[0]=0
                        bar.progress(0,text='Đang bắt đầu lại bằng CPU…')
                        result=engine.predict(raw,clinical,with_cam=with_cam,progress=update_progress)
                    result['processing_device']=str(engine.device)
                    bar.empty()
                result['elapsed_seconds']=time.perf_counter()-started
                result['input_signature']=signature
                probs=result['probs'];pred=CLASSES[result['pred']]
                result.update(patient_id=patient_id,clinical_dict=clinical,raw_slices=raw,model_name=E.MODEL_NAME,
                              pred_class=pred,risk_info=PP.calculate_risk_score(probs),gate_mean=result['gate'])
                st.session_state['pred_result']=result
                st.session_state.pop('exported_report',None)
                try:
                    record_id=H.save_prediction_snapshot(result,filename)
                    result['record_id']=record_id
                    st.success('Đã tổng hợp đủ 15 checkpoint và lưu ảnh cùng kết quả vào Lịch sử ca.')
                except Exception as save_error:
                    st.warning('Dự đoán đã hoàn tất nhưng chưa lưu được hồ sơ: '+str(save_error))
            except Exception as exc:
                st.error('Không thể dự đoán: '+str(exc))
    res=st.session_state.get('pred_result')
    if res is not None:
        st.subheader('3. Kết quả phân loại')
        st.success(f"{res['pred_class']} — {CLASS_LABELS[res['pred_class']]}")
        a,b=st.columns(2)
        a.metric('Xác suất lớp được dự đoán',f"{100*float(np.max(res['probs'])):.1f}%")
        b.metric('Thời gian xử lý',f"{res.get('elapsed_seconds',0):.1f} giây")
        st.caption('Thiết bị đã xử lý: '+res.get('processing_device','Không ghi nhận'))
        ordered=np.sort(res['probs'])
        if ordered[-1]-ordered[-2]<.10:
            st.warning('Hai lớp có xác suất gần nhau (chênh dưới 10 điểm %). Cần thận trọng khi diễn giải kết quả.')
        st.caption('Xác suất do mô hình ước lượng; chưa được xác thực là độ tin cậy lâm sàng. Đây không phải kết luận chẩn đoán.')
        st.dataframe(pd.DataFrame({'Lớp':CLASSES,'Xác suất (%)':res['probs']*100}),hide_index=True,width='stretch')
        st.bar_chart(pd.DataFrame({'Xác suất (%)':res['probs']*100},index=CLASSES),color='#247B85')
        st.caption('Xem bản đồ tại tab Grad-CAM (nếu đã bật); tải kết quả tại tab Xuất báo cáo.')
        with st.expander('Chi tiết mô hình'):
            st.caption(f"Đã tổng hợp {res['models_averaged']} checkpoint · protocol {res['protocol_id'][:12]}")

with tabs[2]:
    st.subheader('Test hold-out OASIS: 39 người')
    saved=read_result('independent_test_results.json')
    values=saved['metrics']
    st.dataframe(pd.DataFrame({'Chỉ số':['Accuracy','Macro-F1','Balanced Accuracy','AUC OvR'],
                              'Giá trị (%)':[100*values[k] for k in ['accuracy','macro_f1','balanced_accuracy','auc_ovr']]}),hide_index=True,width='stretch')
    st.caption('Kết quả đã lưu của cùng ensemble 15 checkpoint. Đây là điểm test, không phải trung bình validation và không gán cho một checkpoint riêng.')
    st.dataframe(pd.DataFrame(values['confusion_matrix'],index=CLASSES,columns=CLASSES),width='stretch')
    st.caption('Ma trận nhầm lẫn: hàng là nhãn tham chiếu, cột là lớp dự đoán. Các ô trên đường chéo là số ca phân loại đúng.')
    if (E.OUTPUT/'adni_evaluation/adni_holdout_results.json').exists():
        st.caption('ADNI: 37 người, Macro-F1 28,57%; AUC 64,83%. Thiếu bốn đặc trưng và đoàn hệ từng được dùng trước đây; kết quả mang tính thăm dò.')

with tabs[3]:
    res=st.session_state.get('pred_result')
    if res is None or res['cams'] is None:
        st.info('Bật Grad-CAM tại mục Dự đoán rồi thực hiện dự đoán.')
    else:
        st.write(res['cam_source'])
        st.caption('Lớp mục tiêu: '+CLASSES[res['target']]+'. Xác suất hiển thị là của ensemble; bản đồ chỉ từ một thành viên. Chưa xác thực độ trung thực hoặc vị trí tổn thương.')
        fig=viz.cam_grid(res['raw_slices'],res['cams'],['Axial','Coronal','Sagittal'])
        st.pyplot(fig)
        plt.close(fig)
        with st.expander('Xem rõ từng lát cắt và thang màu'):
            plane=st.selectbox('Mặt phẳng',['Axial','Coronal','Sagittal'])
            slice_no=st.slider('Lát cắt trong mặt phẳng',1,9,5)
            opacity=st.slider('Độ đậm bản đồ nhiệt',0.,1.,.45,.05)
            index=['Axial','Coronal','Sagittal'].index(plane)*9+slice_no-1
            detail,axes=plt.subplots(1,2,figsize=(9,4))
            for ax in axes:ax.imshow(res['raw_slices'][index],cmap='gray');ax.axis('off')
            axes[0].set_title('MRI');axes[1].set_title('Grad-CAM')
            heat=axes[1].imshow(res['cams'][index].squeeze(),cmap='jet',alpha=opacity,vmin=0,vmax=1,
                extent=axes[1].images[0].get_extent())
            detail.colorbar(heat,ax=axes[1],label='Mức ảnh hưởng tương đối (0–1)',shrink=.7)
            st.pyplot(detail);plt.close(detail)
            st.caption('Màu nóng thể hiện mức ảnh hưởng lớn hơn sau chuẩn hóa. Bản đồ không phải mask phân đoạn hồi hải mã.')

with tabs[4]:
    res=st.session_state.get('pred_result')
    if res is None:
        st.info('Thực hiện dự đoán trước khi xuất báo cáo.')
    else:
        notes=st.text_area('Ghi chú',key='pred_doc_notes')
        provenance='Trung bình đều 15 checkpoint hold-out; protocol '+res['protocol_id']+'. '
        if res['cam_source']:provenance+='Grad-CAM minh họa: '+res['cam_source']+'. '
        args=dict(patient_id=res['patient_id'],clinical_info=res['clinical_dict'],model_name=res['model_name'],
                  pred_class=res['pred_class'],probs=res['probs'],risk_info=res['risk_info'],gate_val=res['gate_mean'],notes=provenance+notes)
        if st.button('Tạo báo cáo',key='make_report'):
            try:
                with st.spinner('Đang tạo báo cáo…'):
                    st.session_state['exported_report']=(R.build_custom_prediction_docx(**args),R.build_custom_prediction_pdf(**args),res['patient_id'])
                    st.session_state['exported_notes']=notes
            except Exception as exc:
                st.error('Chưa thể tạo báo cáo: '+str(exc))
        exported=st.session_state.get('exported_report')
        if exported and st.session_state.get('exported_notes')!=notes:
            st.info('Ghi chú đã thay đổi. Bấm Tạo báo cáo để cập nhật file tải xuống.')
            exported=None
        if exported and exported[2]==res['patient_id']:
            safe_id=re.sub(r'[^\w-]','_',res['patient_id'])[:60] or 'ca'
            st.download_button('Tải Word',exported[0],file_name=f'Ket_qua_{safe_id}.docx',mime='application/vnd.openxmlformats-officedocument.wordprocessingml.document')
            st.download_button('Tải PDF',exported[1],file_name=f'Ket_qua_{safe_id}.pdf',mime='application/pdf')

with tabs[5]:
    st.subheader('Lịch sử ảnh MRI và kết quả dự đoán')
    st.caption('Hồ sơ lưu trên máy này, mỗi lần dự đoán là một lượt riêng. Ảnh lưu là 27 lát đầu vào đã xử lý; kết quả là dự đoán của ca, không phải chỉ số đánh giá toàn bộ mô hình.')
    records=H.list_prediction_snapshots()
    query=st.text_input('Tìm theo mã ca',key='history_search').strip().casefold()
    records=[r for r in records if query in str(r.get('patient_id','')).casefold()]
    if not records:
        st.info('Chưa có hồ sơ phù hợp. Các lần dự đoán mới sẽ được lưu ở đây cùng ảnh MRI. Lịch sử CSV cũ không có ảnh nên không khôi phục được ảnh từ các dòng đó.')
    else:
        labels={r['record_id']:f"{r['patient_id']} · {r['saved_at'].replace('T',' ')} · {r['pred_class']} · {r['filename']}" for r in records}
        selected=st.selectbox('Chọn lượt khám để xem lại',list(labels),format_func=labels.get,key='history_record')
        try:
            record=H.load_prediction_snapshot(selected)
            st.write(f"**Mã ca:** {record['patient_id']} · **Thời gian:** {record['saved_at'].replace('T',' ')}")
            st.info(f"Kết quả đã lưu: {record['pred_class']} — {CLASS_LABELS[record['pred_class']]}")
            st.bar_chart(pd.DataFrame({'Xác suất (%)':record['probs']*100},index=CLASSES),color='#247B85')
            st.dataframe(pd.DataFrame([record['clinical_dict']]).rename(columns=PP.FEATURE_LABELS_VI),hide_index=True)
            plane=st.selectbox('Mặt phẳng MRI',['Axial','Coronal','Sagittal'],key='history_plane')
            number=st.slider('Lát cắt',1,9,5,key='history_slice')
            idx=['Axial','Coronal','Sagittal'].index(plane)*9+number-1
            fig,ax=plt.subplots(figsize=(5,5))
            ax.imshow(record['raw_slices'][idx],cmap='gray');ax.axis('off');ax.set_title(f'{plane} · {number}/9')
            if record['cams'] is not None and st.checkbox('Chồng Grad-CAM đã lưu',key='history_cam'):
                heat=ax.imshow(record['cams'][idx].squeeze(),cmap='jet',alpha=.45,vmin=0,vmax=1,extent=ax.images[0].get_extent())
                fig.colorbar(heat,ax=ax,shrink=.7,label='Mức ảnh hưởng tương đối')
                st.caption(record.get('cam_source','Grad-CAM minh họa'))
            st.pyplot(fig);plt.close(fig)
            st.caption('Mô hình: '+record['model_name'])
            st.download_button('Tải thông tin kết quả (JSON)',json.dumps({k:v for k,v in record.items() if k not in ['raw_slices','cams','probs']},ensure_ascii=False,indent=2),file_name=f'ho_so_{selected}.json',mime='application/json')
        except Exception as exc:
            st.error('Không mở được hồ sơ đã lưu: '+str(exc))
