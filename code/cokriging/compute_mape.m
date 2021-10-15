function [ val1,val2,val3,cok1,cok2,cok3,kg1,kg2,kg3 ] = compute_mape( train_n,val_n,r)
%COMPUTE_MAPE �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
rand_num=1;
addpath('dace');
Ms = csvread('./data/mysqlResult_transfer1.csv', 1, 0);
Mx = csvread('./data/mysqlResult_transfer2.csv', 1, 0);
Mtrain = csvread('./data/train.csv');
Mval = csvread('./data/val.csv');

temp_sum=ones(rand_num,9);
for i = 1:rand_num
    size_mx=size(Mx)
    all_num=size_mx(1)
    feature_size=size_mx(2)-1
    label_pos=size_mx(2)
    
    idx = randperm(all_num);%���������������
    train_idx=idx(1:train_n);%ȡǰn��
   % val_idx=idx((train_n+1):(train_n+val_n))
    val_idx=idx(all_num-val_n:all_num)
    Mtrain=Mx(train_idx,:);%ѵ����
    Mval = Mx(val_idx,:);%���Լ�
    
    Mval=sortrows(Mval,label_pos)
    
    Ms_x=Ms(1:r,1:10);
    Ms_y=Ms(1:r,11);
    Mtrain_x=Mtrain(:,1:feature_size);
    Mtrain_y=Mtrain(:,label_pos);
    Mval_x=Mval(:,1:feature_size);
    Mval_y=Mval(:,label_pos);

    [dmodel,dmc,dmd]=cokriging2(Mtrain_x,Mtrain_y,Ms_x,Ms_y,@regpoly0,@corrgauss,1e-6,1e2);
    cok=predict_cok2(Mval_x,dmodel);
    %co_k=sum(abs(c-Mval_y))/(454-n)
    %meap = mean(abs((observed - predicted)./observed))*100
    %meap_co=mean(abs((Mval_y-cok)./Mval_y))*100;
    temp_sum(i,1)=Mval_y(1);
    temp_sum(i,2)=Mval_y(100);
    temp_sum(i,3)=Mval_y(200);
    temp_sum(i,4)=cok(1);
    temp_sum(i,5)=cok(100);
    temp_sum(i,6)=cok(200);

    krig = dacefit(Mtrain_x,Mtrain_y,@regpoly0,@corrgauss,1e-6,1e-3,20);
    kg = predictor(Mval_x,krig);
    %meap_k=mean(abs((Mval_y-kg)./Mval_y))*100;
    
    temp_sum(i,7)=kg(1);
    temp_sum(i,8)=kg(100);
    temp_sum(i,9)=kg(200);
    
    %temp_sum(i,1)=meap_co;
    %temp_sum(i,2)=meap_k;

end
%co=mean(temp_sum(:,1));
%k=mean(temp_sum(:,2));
val1=mean(temp_sum(:,1));
val2=mean(temp_sum(:,2));
val3=mean(temp_sum(:,3));
cok1=mean(temp_sum(:,4));
cok2=mean(temp_sum(:,5));
cok3=mean(temp_sum(:,6));
kg1=mean(temp_sum(:,7));
kg2=mean(temp_sum(:,8));
kg3=mean(temp_sum(:,9));
end

