function [ co,k] = bin_compute_mape( train_n,ref_idx_1,ref_idx_2,train_idx_1,train_idx_2,val_idx_1,val_idx_2)
%COMPUTE_MAPE �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
rand_num=20;
addpath('dace');
Ms = csvread('./data/mysqlResult_transfer1.csv', 1, 0);
% Mx = csvread('./data/mysqlResult_transfer2.csv', 1, 0);
Mx=csvread('./data/new_train.csv',1,0)
Mval = csvread('./data/new_tw_200_val.csv',1,0);

all_num=size(Mx,1)
all_feature=size(Mx,2)
feature_size=all_feature-1
label_pos=all_feature
Mval=sortrows(Mval,label_pos+1)

Ms1=Ms(ref_idx_1,:)
Ms2=Ms(ref_idx_2,:)
Mx1=Mx(train_idx_1,:)
Mx2=Mx(train_idx_2,:)
Mval1=Mval(val_idx_1,:)
Mval2=Mval(val_idx_2,:)


temp_sum=ones(rand_num,2);
for i = 1:rand_num
%     �ֳ���·���ֱ�ѵ���������temp�ǲ���
    idx = randperm(size(Mx1,1));%���������������
    train_idx=idx(1:train_n/2);%ȡǰn��
   % val_idx=idx((train_n+1):(train_n+val_n))
%     val_idx=idx(all_num-val_n:all_num)
    Mtrain=Mx1(train_idx,:);%ѵ����
%     Mval = Mx(val_idx,:);%���Լ�
    
    Ms_x=Ms1(:,1:10);
    Ms_y=Ms1(:,11);
    Mtrain_x=Mtrain(:,1:feature_size);
    Mtrain_y=Mtrain(:,label_pos);
    Mval_x=Mval1(:,1:feature_size);
    Mval_y=Mval1(:,label_pos+1);
    Mval_ori_y=Mval1(:,label_pos);
    temp_cok=Mval_y
    temp_kg=Mval_y
    [dmodel,dmc,dmd]=cokriging2(Ms_x,Ms_y,Mtrain_x,Mtrain_y,@regpoly0,@corrspline,1e-6,1e2);
    cok1=predict_cok2(Mval_x,dmodel);
    
    krig = dacefit(Mtrain_x,Mtrain_y,@regpoly0,@corrspline,1e-6,1e-3,20);
    kg1 = predictor(Mval_x,krig);

    
    %     �ֳ���·���ֱ�ѵ���������temp�ǲ���
    idx = randperm(size(Mx2,1));%���������������
    train_idx=idx(1:train_n/2);%ȡǰn��
   % val_idx=idx((train_n+1):(train_n+val_n))
%     val_idx=idx(all_num-val_n:all_num)
    Mtrain=Mx2(train_idx,:);%ѵ����
%     Mval = Mx(val_idx,:);%���Լ�
    Ms_x=Ms2(:,1:10);
    Ms_y=Ms2(:,11);
    Mtrain_x=Mtrain(:,1:feature_size);
    Mtrain_y=Mtrain(:,label_pos);
    Mval_x=Mval2(:,1:feature_size);
    Mval_y=Mval(:,label_pos+1);
    Mval_ori_y=Mval(:,label_pos);
    temp_cok=Mval_y
    temp_kg=Mval_y
    
    [dmodel,dmc,dmd]=cokriging2(Ms_x,Ms_y,Mtrain_x,Mtrain_y,@regpoly0,@corrspline,1e-6,1e2);
    cok2=predict_cok2(Mval_x,dmodel);
    temp_cok(val_idx_1,1)=cok1
    temp_cok(val_idx_2,1)=cok2
    
    krig = dacefit(Mtrain_x,Mtrain_y,@regpoly0,@corrspline,1e-6,1e-3,20);
    kg2 = predictor(Mval_x,krig);
    temp_kg(val_idx_1,1)=kg1
    temp_kg(val_idx_2,1)=kg2
    
    meap_co=mean(abs((Mval_y-temp_cok)./Mval_y))*100;
%     temp_sum(i,1)=Mval_y(1);
%     temp_sum(i,2)=Mval_y(100);
%     temp_sum(i,3)=Mval_y(200);
%     temp_sum(i,4)=cok(1);
%     temp_sum(i,5)=cok(100);
%     temp_sum(i,6)=cok(200);

    meap_k=mean(abs((Mval_y-temp_kg)./Mval_y))*100;
    
%     temp_sum(i,7)=kg(1);
%     temp_sum(i,8)=kg(100);
%     temp_sum(i,9)=kg(200);
    
    temp_sum(i,1)=meap_co;
    temp_sum(i,2)=meap_k;
    
    if i==1
        various={'co_kg','Mval_y','Mval_ori_y'};
        %�������
        result_table=table(temp_cok(:,1),Mval_y(:,1),Mval_ori_y(:,1),'VariableNames',various);
        %����csv���
        writetable(result_table, [num2str(train_n),'_cok&val_y_ori&bin.csv'])
    end
end
co=mean(temp_sum(:,1));
k=mean(temp_sum(:,2));
% val1=mean(temp_sum(:,1));
% val2=mean(temp_sum(:,2));
% val3=mean(temp_sum(:,3));
% cok1=mean(temp_sum(:,4));
% cok2=mean(temp_sum(:,5));
% cok3=mean(temp_sum(:,6));
% kg1=mean(temp_sum(:,7));
% kg2=mean(temp_sum(:,8));
% kg3=mean(temp_sum(:,9));
end

