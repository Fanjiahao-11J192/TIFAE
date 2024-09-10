import torch
import os
import json
from collections import OrderedDict
import torch.nn.functional as F
from models.base_model import BaseModel
from models.networks.fc import FcEncoder
from models.networks.lstm import LSTMEncoder
from models.networks.textcnn import TextCNN
from models.networks.classifier import FcClassifier,Fusion
from models.networks.autoencoder2 import ResidualAE
from models.utt_fusion_model import UttFusionModel
from models.networks.Transformer.transformer import TransformerEncoder
from .utils.config import OptConfig
from torch import nn
from models.networks.ModalEnhancedModule import MEM,MEM2
class MTMMIN5Model(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--input_dim_a', type=int, default=130, help='acoustic input dim')
        parser.add_argument('--input_dim_l', type=int, default=1024, help='lexical input dim')
        parser.add_argument('--input_dim_v', type=int, default=384, help='lexical input dim')
        parser.add_argument('--embd_size_a', default=128, type=int, help='audio model embedding size')
        parser.add_argument('--embd_size_l', default=128, type=int, help='text model embedding size')
        parser.add_argument('--embd_size_v', default=128, type=int, help='visual model embedding size')
        parser.add_argument('--embd_method_a', default='maxpool', type=str, choices=['last', 'maxpool', 'attention'],
                            help='audio embedding method,last,mean or atten')
        parser.add_argument('--embd_method_v', default='maxpool', type=str, choices=['last', 'maxpool', 'attention'],
                            help='visual embedding method,last,mean or atten')
        parser.add_argument('--AE_layers', type=str, default='128,64,32',
                            help='256,128 for 2 layers with 256, 128 nodes respectively')
        parser.add_argument('--n_blocks', type=int, default=3, help='number of AE blocks')
        parser.add_argument('--cls_layers', type=str, default='128,128',
                            help='256,128 for 2 layers with 256, 128 nodes respectively')
        parser.add_argument('--dropout_rate', type=float, default=0.3, help='rate of dropout')
        parser.add_argument('--bn', action='store_true', help='if specified, use bn layers in FC')
        parser.add_argument('--pretrained_path', type=str, help='where to load pretrained encoder network')
        parser.add_argument('--ce_weight', type=float, default=1.0, help='weight of ce loss')
        parser.add_argument('--mse_weight', type=float, default=1.0, help='weight of mse loss')
        parser.add_argument('--cycle_weight', type=float, default=1.0, help='weight of cycle loss')
        parser.add_argument('--trans_weight', type=float, default=1.0, help='weight of transformer loss')
        parser.add_argument('--share_weight', action='store_true',
                            help='share weight of forward and backward autoencoders')
        parser.add_argument('--transformer_heads', default=4, type=int, help='transformer multimodal head numbers')
        parser.add_argument('--transformer_layers', default=6, type=int, help='transformer encode layers numbers')
        parser.add_argument('--conv_dim_a', default=40, type=int, help='acoustic conv dim')
        parser.add_argument('--conv_dim_v', default=40, type=int, help='visual conv dim')
        parser.add_argument('--conv_dim_l', default=40, type=int, help='lexical conv dim')
        parser.add_argument('--image_dir', type=str, default='./shared_image', help='models image are saved here')
        return parser

    def __init__(self, opt):
        """Initialize the LSTM autoencoder class
        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        super().__init__(opt)
        # our expriment is on 10 fold setting, teacher is on 5 fold setting, the train set should match
        #self.loss_names = ['CE', 'mse', 'cycle']
        #self.loss_names = ['CE', 'mse']
        self.loss_names = ['CE', 'mse', 'trans']
        self.model_names = ['A', 'V', 'L', 'C', 'AE']

        # acoustic model
        self.netA = LSTMEncoder(opt.input_dim_a, opt.embd_size_a, embd_method=opt.embd_method_a)
        # lexical model
        self.netL = TextCNN(opt.input_dim_l, opt.embd_size_l)
        # visual model
        self.netV = LSTMEncoder(opt.input_dim_v, opt.embd_size_v, opt.embd_method_v)

       

        self.model_names.append('AL_transformerEncoder')
        self.netAL_transformerEncoder = TransformerEncoder(opt.conv_dim_l, opt.transformer_heads,
                                                           opt.transformer_layers).to(self.device)

        self.model_names.append('VL_transformerEncoder')
        self.netVL_transformerEncoder = TransformerEncoder(opt.conv_dim_l, opt.transformer_heads,
                                                           opt.transformer_layers).to(self.device)

        self.model_names.append('AV_transformerEncoder')
        self.netAV_transformerEncoder = TransformerEncoder(opt.conv_dim_v, opt.transformer_heads,
                                                           opt.transformer_layers).to(self.device)

        self.model_names.append('LV_transformerEncoder')
        self.netLV_transformerEncoder = TransformerEncoder(opt.conv_dim_v, opt.transformer_heads,
                                                           opt.transformer_layers).to(self.device)

        self.model_names.append('VA_transformerEncoder')
        self.netVA_transformerEncoder = TransformerEncoder(opt.conv_dim_a, opt.transformer_heads,
                                                           opt.transformer_layers).to(self.device)

        self.model_names.append('LA_transformerEncoder')
        self.netLA_transformerEncoder = TransformerEncoder(opt.conv_dim_a, opt.transformer_heads,
                                                           opt.transformer_layers).to(self.device)

        self.model_names.append('A_transformerEncoder')
        self.netA_transformerEncoder = TransformerEncoder(opt.conv_dim_a * 2, opt.transformer_heads,
                                                          opt.transformer_layers).to(self.device)

        self.model_names.append('V_transformerEncoder')
        self.netV_transformerEncoder = TransformerEncoder(opt.conv_dim_v * 2, opt.transformer_heads,
                                                          opt.transformer_layers).to(self.device)

        self.model_names.append('L_transformerEncoder')
        self.netL_transformerEncoder = TransformerEncoder(opt.conv_dim_l * 2, opt.transformer_heads,
                                                          opt.transformer_layers).to(self.device)
        # 加入一层卷积层
        self.model_names.append('L_Conv')
        self.netL_Conv = nn.Conv1d(opt.embd_size_l, opt.conv_dim_l, kernel_size=3, padding=0, bias=False)
        self.model_names.append('A_Conv')
        self.netA_Conv = nn.Conv1d(opt.embd_size_a, opt.conv_dim_a, kernel_size=3, padding=0, bias=False)
        self.model_names.append('V_Conv')
        self.netV_Conv = nn.Conv1d(opt.embd_size_v, opt.conv_dim_v, kernel_size=3, padding=0, bias=False)

        # AE model
        AE_layers = list(map(lambda x: int(x), opt.AE_layers.split(',')))
        AE_input_dim = 2*(opt.conv_dim_l+opt.conv_dim_a+opt.conv_dim_v)+(opt.embd_size_a + opt.embd_size_v + opt.embd_size_l)
        AE_output_dim = opt.embd_size_a + opt.embd_size_v + opt.embd_size_l
        self.netAE = ResidualAE(AE_layers, opt.n_blocks, AE_input_dim,AE_output_dim,dropout=0, use_bn=False)
        
        cls_layers = list(map(lambda x: int(x), opt.cls_layers.split(',')))
        cls_input_size = AE_layers[-1] * opt.n_blocks
        if self.opt.corpus_name != 'MOSI':
            self.netC = FcClassifier(cls_input_size, cls_layers, output_dim=opt.output_dim, dropout=opt.dropout_rate,use_bn=opt.bn)
        else:
            self.netC = Fusion(cls_input_size, cls_layers, output_dim=opt.output_dim, dropout=opt.dropout_rate)
        self.A_EM_L = MEM(opt.conv_dim_a,1,1).to(self.device)
        self.A_EM_V = MEM(opt.conv_dim_a, 1, 1).to(self.device)
        self.V_EM_A = MEM(opt.conv_dim_v,1,1).to(self.device)
        self.V_EM_L = MEM(opt.conv_dim_v, 1, 1).to(self.device)
        self.L_EM_A = MEM(opt.conv_dim_l, 1, 1).to(self.device)
        self.L_EM_V = MEM(opt.conv_dim_l, 1, 1).to(self.device)
        if self.isTrain:
            self.load_pretrained_encoder(opt)
            if self.opt.corpus_name != 'MOSI':
                self.criterion_ce = torch.nn.CrossEntropyLoss()
            else:
                self.criterion_ce = torch.nn.MSELoss()
            self.criterion_mse = torch.nn.MSELoss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            paremeters = [{'params': getattr(self, 'net' + net).parameters()} for net in self.model_names]
            self.optimizer = torch.optim.Adam(paremeters, lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer)
            self.output_dim = opt.output_dim
            self.ce_weight = opt.ce_weight
            self.mse_weight = opt.mse_weight
            self.cycle_weight = opt.cycle_weight
            self.trans_weight = opt.trans_weight
        else:
            self.load_pretrained_encoder(opt)

        # modify save_dir
        self.save_dir = os.path.join(self.save_dir, str(opt.cvNo))
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

        image_save_dir = os.path.join(opt.image_dir, opt.name)
        image_save_dir = os.path.join(image_save_dir, str(opt.cvNo))
        self.predict_image_save_dir = os.path.join(image_save_dir, 'predict')
        self.trans_image_save_dir = os.path.join(image_save_dir, 'trans')
        self.loss_image_save_dir = os.path.join(image_save_dir, 'loss')
        if not os.path.exists(self.predict_image_save_dir):
            os.makedirs(self.predict_image_save_dir)
        if not os.path.exists(self.trans_image_save_dir):
            os.makedirs(self.trans_image_save_dir)
        if not os.path.exists(self.loss_image_save_dir):
            os.makedirs(self.loss_image_save_dir)

    def load_pretrained_encoder(self, opt):
        print('Init parameter from {}'.format(opt.pretrained_path))
        pretrained_path = os.path.join(opt.pretrained_path, str(opt.cvNo))
        pretrained_config_path = os.path.join(opt.pretrained_path, 'train_opt.conf')
        pretrained_config = self.load_from_opt_record(pretrained_config_path)
        pretrained_config.isTrain = False  # teacher model should be in test mode
        pretrained_config.gpu_ids = opt.gpu_ids  # set gpu to the same
        self.pretrained_encoder = UttFusionModel(pretrained_config)
        self.pretrained_encoder.load_networks_cv(pretrained_path)
        self.pretrained_encoder.cuda()
        self.pretrained_encoder.eval()

    def post_process(self):
        # called after model.setup()
        def transform_key_for_parallel(state_dict):
            return OrderedDict([('module.' + key, value) for key, value in state_dict.items()])

        if self.isTrain:
            print('[ Init ] Load parameters from pretrained encoder network')
            f = lambda x: transform_key_for_parallel(x)
            self.netA.load_state_dict(f(self.pretrained_encoder.netA.state_dict()))
            self.netV.load_state_dict(f(self.pretrained_encoder.netV.state_dict()))
            self.netL.load_state_dict(f(self.pretrained_encoder.netL.state_dict()))

            self.netA_Conv.load_state_dict(f(self.pretrained_encoder.netA_Conv.state_dict()))
            self.netV_Conv.load_state_dict(f(self.pretrained_encoder.netV_Conv.state_dict()))
            self.netL_Conv.load_state_dict(f(self.pretrained_encoder.netL_Conv.state_dict()))

            self.netAL_transformerEncoder.load_state_dict(f(self.pretrained_encoder.netAL_transformerEncoder.state_dict()))
            self.netVL_transformerEncoder.load_state_dict(f(self.pretrained_encoder.netVL_transformerEncoder.state_dict()))
            self.netL_transformerEncoder.load_state_dict(f(self.pretrained_encoder.netL_transformerEncoder.state_dict()))

            self.netAV_transformerEncoder.load_state_dict(f(self.pretrained_encoder.netAV_transformerEncoder.state_dict()))
            self.netLV_transformerEncoder.load_state_dict(f(self.pretrained_encoder.netLV_transformerEncoder.state_dict()))
            self.netV_transformerEncoder.load_state_dict(f(self.pretrained_encoder.netV_transformerEncoder.state_dict()))

            self.netVA_transformerEncoder.load_state_dict(f(self.pretrained_encoder.netVA_transformerEncoder.state_dict()))
            self.netLA_transformerEncoder.load_state_dict(f(self.pretrained_encoder.netLA_transformerEncoder.state_dict()))
            self.netA_transformerEncoder.load_state_dict(f(self.pretrained_encoder.netA_transformerEncoder.state_dict()))
    def load_from_opt_record(self, file_path):
        opt_content = json.load(open(file_path, 'r'))
        opt = OptConfig()
        opt.load(opt_content)
        return opt

    def set_input(self, input):
        """
        Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        """
        acoustic = input['A_feat'].float().to(self.device)
        lexical = input['L_feat'].float().to(self.device)
        visual = input['V_feat'].float().to(self.device)
        self.acoustic = acoustic
        self.lexical = lexical
        self.visual = visual

        if self.isTrain:
            self.label = input['label'].to(self.device)
            self.missing_index = input['missing_index'].long().to(self.device)
            # A modality
            self.A_miss_index = self.missing_index[:, 0].unsqueeze(1).unsqueeze(2)
            self.A_miss = acoustic * self.A_miss_index
            self.A_reverse = acoustic * -1 * (self.A_miss_index - 1)
            # L modality
            self.L_miss_index = self.missing_index[:, 2].unsqueeze(1).unsqueeze(2)
            self.L_miss = lexical * self.L_miss_index
            self.L_reverse = lexical * -1 * (self.L_miss_index - 1)
            # V modality
            self.V_miss_index = self.missing_index[:, 1].unsqueeze(1).unsqueeze(2)
            self.V_miss = visual * self.V_miss_index
            self.V_reverse = visual * -1 * (self.V_miss_index - 1)
            if self.opt.corpus_name == 'MOSI':
                self.label = self.label.unsqueeze(1)
        else:
            self.A_miss = acoustic
            self.V_miss = visual
            self.L_miss = lexical

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        # get utt level representattion
        self.feat_A_miss,  self.feat_A_miss_noMaxpool = self.netA(self.A_miss)
        self.feat_L_miss,  self.feat_L_miss_noMaxpool = self.netL(self.L_miss)
        self.feat_V_miss,  self.feat_V_miss_noMaxpool = self.netV(self.V_miss)
        # fusion miss
        self.feat_fusion_miss = torch.cat([self.feat_A_miss, self.feat_L_miss, self.feat_V_miss], dim=-1)

        self.feat_A_miss_NoMaxPool = self.feat_A_miss_noMaxpool.transpose(1, 2)  # [batch_size, n_features，seq_len]
        self.feat_V_miss_NoMaxPool = self.feat_V_miss_noMaxpool.transpose(1, 2)  # [batch_size, n_features，seq_len]
        self.feat_L_miss_NoMaxPool = self.feat_L_miss_noMaxpool.transpose(1, 2)  # [batch_size, n_features，seq_len]

        self.feat_A_miss_NoMaxPool = self.netA_Conv(self.feat_A_miss_NoMaxPool)  # [batch_size, 40,seq_len]
        self.feat_V_miss_NoMaxPool = self.netV_Conv(self.feat_V_miss_NoMaxPool)  # [batch_size, 40,seq_len]
        self.feat_L_miss_NoMaxPool = self.netL_Conv(self.feat_L_miss_NoMaxPool)  # [batch_size, 40,seq_len]

        self.feat_A_miss_NoMaxPool = self.feat_A_miss_NoMaxPool.permute(2, 0, 1)  # [seq_len,batch_size, 40]
        self.feat_V_miss_NoMaxPool = self.feat_V_miss_NoMaxPool.permute(2, 0, 1)  # [seq_len,batch_size, 40]
        self.feat_L_miss_NoMaxPool = self.feat_L_miss_NoMaxPool.permute(2, 0, 1)  # [seq_len,batch_size, 40]

        # 执行模态增强操作
        self.feat_A_miss_enhanced_by_L = self.A_EM_L(self.feat_A_miss_NoMaxPool,self.feat_L_miss_NoMaxPool)
        self.feat_A_miss_enhanced_by_V = self.A_EM_V(self.feat_A_miss_NoMaxPool, self.feat_V_miss_NoMaxPool)
        self.feat_V_miss_enhanced_by_L = self.V_EM_L(self.feat_V_miss_NoMaxPool,self.feat_L_miss_NoMaxPool)
        self.feat_V_miss_enhanced_by_A = self.V_EM_A(self.feat_V_miss_NoMaxPool, self.feat_A_miss_NoMaxPool)
        self.feat_L_miss_enhanced_by_A = self.L_EM_A(self.feat_L_miss_NoMaxPool, self.feat_A_miss_NoMaxPool)
        self.feat_L_miss_enhanced_by_V = self.L_EM_V(self.feat_L_miss_NoMaxPool, self.feat_V_miss_NoMaxPool)


    
        '''
            将Transformer-Encoder的Q输入换成原始数据。
        '''

        # 执行跨模态注意力
        self.AL_miss = self.netAL_transformerEncoder(self.feat_L_miss_NoMaxPool, self.feat_A_miss_enhanced_by_V, self.feat_A_miss_enhanced_by_V)
        self.VL_miss = self.netVL_transformerEncoder(self.feat_L_miss_NoMaxPool, self.feat_V_miss_enhanced_by_A, self.feat_V_miss_enhanced_by_A)
        # 执行自注意力
        self.ALVL_miss = self.netL_transformerEncoder(torch.cat([self.AL_miss, self.VL_miss], dim=-1))
        # 执行池化操作
        # self.L = self.L.permute(1,2,0)
        # self.final_L = F.avg_pool1d(self.L, self.L.size(2), self.L.size(2)).squeeze(-1)  # [batch_size,80]
        # 直接取最后一层
        self.final_L_miss = self.ALVL_miss[-1]  # [batch_size,80]

        # 执行跨模态注意力
        self.AV_miss = self.netAV_transformerEncoder(self.feat_V_miss_NoMaxPool, self.feat_A_miss_enhanced_by_L, self.feat_A_miss_enhanced_by_L)
        self.LV_miss = self.netLV_transformerEncoder(self.feat_V_miss_NoMaxPool, self.feat_L_miss_enhanced_by_A, self.feat_L_miss_enhanced_by_A)
        # 执行自注意力
        self.AVLV_miss = self.netV_transformerEncoder(torch.cat([self.AV_miss, self.LV_miss], dim=-1))
        # 执行池化操作
        # self.V = self.V.permute(1, 2, 0)
        # self.final_V = F.avg_pool1d(self.V, self.V.size(2), self.V.size(2)).squeeze(-1)  # [batch_size,40]
        # 直接取最后一层
        self.final_V_miss = self.AVLV_miss[-1]  # [batch_size,40]

        # 执行跨模态注意力
        self.VA_miss = self.netVA_transformerEncoder(self.feat_A_miss_NoMaxPool, self.feat_V_miss_enhanced_by_L, self.feat_V_miss_enhanced_by_L)
        self.LA_miss = self.netLA_transformerEncoder(self.feat_A_miss_NoMaxPool, self.feat_L_miss_enhanced_by_V, self.feat_L_miss_enhanced_by_V)
        # 执行自注意力
        self.VALA_miss = self.netA_transformerEncoder(torch.cat([self.VA_miss, self.LA_miss], dim=-1))
        # 执行池化操作
        # self.A = self.A.permute(1, 2, 0)
        # self.final_A = F.avg_pool1d(self.A, self.A.size(2), self.A.size(2)).squeeze(-1)  # [batch_size,40]
        # 直接取最后一层
        self.final_A_miss = self.VALA_miss[-1]  # [batch_size,40]

        self.crosstransformer_fusion_miss = torch.cat([self.final_A_miss, self.final_L_miss, self.final_V_miss], dim=-1)
        self.final_crosstransformer_fusion_miss = torch.cat([self.crosstransformer_fusion_miss,self.feat_fusion_miss], dim=-1)


        # calc reconstruction of teacher's output
        self.recon_fusion, self.latent = self.netAE(self.final_crosstransformer_fusion_miss)
        
        # get fusion outputs for missing modality
        self.logits, _ = self.netC(self.latent)
        if self.opt.corpus_name != 'MOSI':
            self.pred = F.softmax(self.logits, dim=-1)
        else:
            self.pred = self.logits
        # for training
        if self.isTrain:
            with torch.no_grad():

                self.reverse_A,_ = self.pretrained_encoder.netA(self.A_reverse)
                self.reverse_L, _ = self.pretrained_encoder.netL(self.L_reverse)
                self.reverse_V, _ = self.pretrained_encoder.netV(self.V_reverse)
                self.T_embds_reverse = torch.cat([self.reverse_A, self.reverse_L, self.reverse_V], dim=-1)



                self.T_embd_A, self.T_embd_A_NoMaxPool = self.pretrained_encoder.netA(self.acoustic)
                self.T_embd_L, self.T_embd_L_NoMaxPool = self.pretrained_encoder.netL(self.lexical)
                self.T_embd_V, self.T_embd_V_NoMaxPool = self.pretrained_encoder.netV(self.visual)
                # self.T_embd_A, self.T_embd_A_NoMaxPool = self.pretrained_encoder.netA(self.A_miss)
                # self.T_embd_L, self.T_embd_L_NoMaxPool = self.pretrained_encoder.netL(self.L_miss)
                # self.T_embd_V, self.T_embd_V_NoMaxPool = self.pretrained_encoder.netV(self.V_miss)
                self.feat_fusion_full = torch.cat([self.T_embd_A, self.T_embd_L, self.T_embd_V], dim=-1)


                # embd_A, _ = self.pretrained_encoder.netA(self.acoustic)
                # embd_L, _ = self.pretrained_encoder.netL(self.lexical)
                # embd_V, _ = self.pretrained_encoder.netV(self.visual)
                # self.embds_AVL = torch.cat([embd_A, embd_L, embd_V], dim=-1)


                self.T_embd_A_NoMaxPool = self.T_embd_A_NoMaxPool.transpose(1, 2)  # [batch_size, n_features，seq_len]
                self.T_embd_L_NoMaxPool = self.T_embd_L_NoMaxPool.transpose(1, 2)  # [batch_size, n_features，seq_len]
                self.T_embd_V_NoMaxPool = self.T_embd_V_NoMaxPool.transpose(1, 2)  # [batch_size, n_features，seq_len]

                self.T_embd_A_NoMaxPool = self.pretrained_encoder.netA_Conv(self.T_embd_A_NoMaxPool)  # [batch_size, 40,seq_len]
                self.T_embd_V_NoMaxPool = self.pretrained_encoder.netV_Conv(self.T_embd_V_NoMaxPool)  # [batch_size, 40,seq_len]
                self.T_embd_L_NoMaxPool = self.pretrained_encoder.netL_Conv(self.T_embd_L_NoMaxPool)  # [batch_size, 40,seq_len]

                self.T_embd_A_NoMaxPool = self.T_embd_A_NoMaxPool.permute(2, 0, 1)  # [seq_len,batch_size, 40]
                self.T_embd_V_NoMaxPool = self.T_embd_V_NoMaxPool.permute(2, 0, 1)  # [seq_len,batch_size, 40]
                self.T_embd_L_NoMaxPool = self.T_embd_L_NoMaxPool.permute(2, 0, 1)  # [seq_len,batch_size, 40]

                # 执行跨模态注意力
                self.AL_full = self.pretrained_encoder.netAL_transformerEncoder(self.T_embd_L_NoMaxPool, self.T_embd_A_NoMaxPool,
                                                             self.T_embd_A_NoMaxPool)
                self.VL_full = self.pretrained_encoder.netVL_transformerEncoder(self.T_embd_L_NoMaxPool, self.T_embd_V_NoMaxPool,
                                                             self.T_embd_V_NoMaxPool)
                # 执行自注意力
                self.L_full = self.pretrained_encoder.netL_transformerEncoder(torch.cat([self.AL_full, self.VL_full], dim=-1))
                # 执行池化操作
                # self.L = self.L.permute(1,2,0)
                # self.final_L = F.avg_pool1d(self.L, self.L.size(2), self.L.size(2)).squeeze(-1)  # [batch_size,80]
                # 直接取最后一层
                self.final_L_full = self.L_full[-1]  # [batch_size,80]

                # 执行跨模态注意力
                self.AV_full = self.pretrained_encoder.netAV_transformerEncoder(self.T_embd_V_NoMaxPool, self.T_embd_A_NoMaxPool,
                                                             self.T_embd_A_NoMaxPool)
                self.LV_full = self.pretrained_encoder.netLV_transformerEncoder(self.T_embd_V_NoMaxPool, self.T_embd_L_NoMaxPool,
                                                             self.T_embd_L_NoMaxPool)
                # 执行自注意力
                self.V_full = self.pretrained_encoder.netV_transformerEncoder(torch.cat([self.AV_full, self.LV_full], dim=-1))
                # 执行池化操作
                # self.V = self.V.permute(1, 2, 0)
                # self.final_V = F.avg_pool1d(self.V, self.V.size(2), self.V.size(2)).squeeze(-1)  # [batch_size,40]
                # 直接取最后一层
                self.final_V_full = self.V_full[-1]  # [batch_size,40]

                # 执行跨模态注意力
                self.VA_full = self.pretrained_encoder.netVA_transformerEncoder(self.T_embd_A_NoMaxPool, self.T_embd_V_NoMaxPool,
                                                             self.T_embd_V_NoMaxPool)
                self.LA_full = self.pretrained_encoder.netLA_transformerEncoder(self.T_embd_A_NoMaxPool, self.T_embd_L_NoMaxPool,
                                                             self.T_embd_L_NoMaxPool)
                # 执行自注意力
                self.A_full = self.pretrained_encoder.netA_transformerEncoder(torch.cat([self.VA_full, self.LA_full], dim=-1))
                # 执行池化操作
                # self.A = self.A.permute(1, 2, 0)
                # self.final_A = F.avg_pool1d(self.A, self.A.size(2), self.A.size(2)).squeeze(-1)  # [batch_size,40]
                # 直接取最后一层
                self.final_A_full = self.A_full[-1]  # [batch_size,40]

                self.crosstransformer_fusion_full = torch.cat([self.final_A_full, self.final_L_full, self.final_V_full], dim=-1)
                self.final_crosstransformer_fusion_full = torch.cat([self.crosstransformer_fusion_full, self.feat_fusion_full], dim=-1)


    def backward(self):
        """Calculate the loss for back propagation"""
        self.loss_CE = self.ce_weight * self.criterion_ce(self.logits, self.label)
        
        self.loss_mse = self.mse_weight * self.criterion_mse(self.T_embds_reverse,self.recon_fusion)

        self.loss_trans = self.trans_weight * self.criterion_mse(self.final_crosstransformer_fusion_miss, self.final_crosstransformer_fusion_full)

        loss = self.loss_CE + self.loss_mse + self.loss_trans
        loss.backward()
        for model in self.model_names:
            torch.nn.utils.clip_grad_norm_(getattr(self, 'net' + model).parameters(), 1.0)

    def optimize_parameters(self, epoch):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()
        # backward
        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()
