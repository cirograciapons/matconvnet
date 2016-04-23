%% Experiment with the cnn_mnist_fc_bnorm

%train two networks for regression
[net_bn, info_bn] = regression(...
  'expDir', 'data/regression', 'batchNormalization', true);

[net_fc, info_fc] = regression(...
  'expDir', 'data/regression', 'batchNormalization', false);

figure(1) ; clf ;
subplot(1,2,1) ;
semilogy(cat(1,info_fc.val(:).objective), 'o-') ; hold all ;
semilogy(cat(1,info_bn.val(:).objective)', '+--') ;
xlabel('Training samples [x 10^3]'); ylabel('energy') ;
grid on ;
h=legend('BSLN', 'BNORM') ;
set(h,'color','none');
title('objective') ;
subplot(1,2,2) ;
plot(cat(1,info_fc.val.regressionMse)', 'o-') ; hold all ;
plot(cat(1,info_bn.val.regressionMse)', '+--') ;
h=legend('BSLN-val','BSLN-val-5','BNORM-val','BNORM-val-5') ;
grid on ;
xlabel('Training samples [x 10^3]'); ylabel('error') ;
set(h,'color','none') ;
title('error') ;
drawnow ;