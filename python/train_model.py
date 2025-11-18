'''
| 参数                 | 当前值    | 建议值       | 理由                     |
| ------------------ | ------ | --------- | ---------------------- |
| learning\_rate     | 0.0005 | **0.001** | 模型小，可承受稍大学习率加速收敛       |
| weight\_decay      | 1e-3   | **5e-4**  | 减少正则化强度，避免欠拟合          |
| dropout\_rate      | 0.1    | **0.05**  | 模型参数少，可降低Dropout保留更多特征 |
| confusion\_penalty | 0.8    | **0.5**   | 当前损失1.0+，惩罚项0.8可能过大    |
# 在train()函数中
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, 
    T_max=num_epochs,  # 改为num_epochs，避免过度衰减
    eta_min=1e-5       # 提高最低学习率，防止收敛过慢
)
'''
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import os
from sklearn.preprocessing import StandardScaler
import joblib
from scipy.signal import square
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.signal import sawtooth
# 设置字体以支持中文
plt.rcParams['font.sans-serif'] = ['SimHei'] # 使用黑体
plt.rcParams['axes.unicode_minus'] = False # 正常显示负号
# 设置随机种子保证可重复性
torch.manual_seed(42)
np.random.seed(42)

class EarlyStopping:
    """早停类，用于在验证损失不再下降时停止训练"""
    def __init__(self, patience=50, min_delta=0.001, verbose=True):
        """
        Args:
            patience (int): 在验证损失不再下降后等待的epoch数（增大到50）
            min_delta (float): 被视为改善的最小变化（增大到0.001）
            verbose (bool): 是否打印早停信息
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model_state = model.state_dict().copy()
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'早停计数器: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print('触发早停!')
        else:
            self.best_loss = val_loss
            self.best_model_state = model.state_dict().copy()
            self.counter = 0

class AdaptiveFocusedLoss(nn.Module):
    """固定权重的聚焦损失，不再根据混淆矩阵更新"""
    def __init__(self, num_classes=4, base_weights=None, confusion_penalty=0.5):
        super(AdaptiveFocusedLoss, self).__init__()
        self.num_classes = num_classes
        self.base_loss = nn.CrossEntropyLoss()
        self.confusion_penalty = confusion_penalty

        # 优化后的权重矩阵 - 重点调整方波与三角波之间的混淆
        self.confusion_weights = base_weights

    def forward(self, outputs, labels):
        base_loss = self.base_loss(outputs, labels)

        # 保留"对高概率错类加权惩罚"的代码，但权重固定
        batch_size = outputs.size(0)
        confusion_penalty = 0.0
        count = 0

        with torch.no_grad():          # 惩罚项不参与梯度，仅作数值加权
            for i in range(batch_size):
                true_label = labels[i].item()
                probs = torch.softmax(outputs[i], dim=0)
                for wrong_label in range(self.num_classes):
                    if wrong_label == true_label:
                        continue
                    weight = self.confusion_weights[true_label, wrong_label]
                    if weight > 1.0:
                        wrong_prob = probs[wrong_label].item()
                        if wrong_prob > 0.2:          # 仍可调阈值
                            confusion_penalty += weight * wrong_prob
                            count += 1

        if count > 0:
            confusion_penalty /= count

        total_loss = base_loss + self.confusion_penalty * confusion_penalty
        return total_loss

class SignalDataset(Dataset):
    """自定义信号数据集类"""
    def __init__(self, num_samples=60000, sequence_length=64, include_other=True):
        self.num_samples = num_samples
        self.sequence_length = sequence_length
        self.include_other = include_other
        self.signals = []
        self.labels = []
        self._generate_dataset()
    
    def _generate_pure_noise(self, amplitude, t):
        """生成纯噪声信号"""
        # 生成高斯白噪声
        noise = np.random.normal(0, amplitude * 0.1, len(t))
        return noise
    
    def _generate_third_harmonic_modulation(self, amplitude, frequency, phase, t):
        """生成三次频率调制信号"""
        # 基波 + 三次谐波调制
        fundamental = amplitude * np.sin(2 * np.pi * frequency * t + phase)
        third_harmonic = 0.3 * amplitude * np.sin(2 * np.pi * 3 * frequency * t + 1.5 * phase)
        signal = fundamental + third_harmonic
        return signal
    
    def _generate_local_linear(self, amplitude, t):
        """生成本地线性信号"""
        num_segments = np.random.randint(4, 8)
        segment_length = len(t) // num_segments
        
        signal = np.zeros(len(t))
        current_value = np.random.uniform(-amplitude, amplitude)
        
        for seg in range(num_segments):
            start_idx = seg * segment_length
            end_idx = min((seg + 1) * segment_length, len(t))
            
            # 随机决定下一个值
            next_value = np.random.uniform(-amplitude, amplitude)
            
            # 线性插值
            if end_idx > start_idx:
                for i in range(start_idx, end_idx):
                    alpha = (i - start_idx) / (end_idx - start_idx)
                    signal[i] = current_value * (1 - alpha) + next_value * alpha
            
            current_value = next_value
        
        # 添加平滑过渡
        from scipy import ndimage
        signal = ndimage.gaussian_filter1d(signal, sigma=1.0)
        
        return signal
    
    def _generate_non_periodic_chaos(self, amplitude, t):
        """生成非周期混沌信号"""
        # 使用Logistic映射生成混沌序列
        chaos_param = np.random.uniform(3.7, 4.0)  # 混沌参数
        chaos_seq = np.zeros(len(t))
        chaos_seq[0] = np.random.uniform(0.1, 0.9)
        for i in range(1, len(t)):
            chaos_seq[i] = chaos_param * chaos_seq[i-1] * (1 - chaos_seq[i-1])
        signal = amplitude * 2 * (chaos_seq - 0.5)  # 映射到[-A, A]
        return signal

    def _generate_other_signals(self, amplitude, frequency, phase, t):
        """生成其他信号，按照新的分布"""
        signal_type = np.random.random()
        
        if signal_type < 0.8:  # 50% 纯噪声
            signal = self._generate_pure_noise(amplitude, t)
        elif signal_type < 0.81:  # 10% 三次频率调制
            signal = self._generate_third_harmonic_modulation(amplitude, frequency, phase, t)
        elif signal_type < 0.92:  # 20% 局部线性
            signal = self._generate_local_linear(amplitude, t)
        else:  # 20% 非周期混沌
            signal = self._generate_non_periodic_chaos(amplitude, t)
        
        # 确保信号幅值在[-5, 5]范围内
        if np.max(np.abs(signal)) > 5.0:
            signal = signal * (5.0 / np.max(np.abs(signal)))
        
        return signal

    def _generate_dataset(self):
        """生成训练数据集"""
        print("正在生成信号数据集...")
        
        for i in range(self.num_samples):
            # 决定是否为纯信号 (60%概率)
            is_signal = np.random.random() < 0.8
            
            if is_signal:
                # 纯信号：从正弦波、方波、三角波中随机选择
                signal_type = np.random.randint(0, 3)
                # 纯信号的噪声水平大幅减少
                noise_level = np.random.uniform(0, 0.05)
            else:
                # 20%概率为其他信号
                signal_type = 3
                # 其他信号的噪声水平也减少
                noise_level = np.random.uniform(0.1, 0.5)
            
            # 生成随机参数
            amplitude = np.random.uniform(1.5, 3.5)
            frequency = 1
            phase = np.random.uniform(0, 2*np.pi)               

            # 生成时间序列
            num_cycles = np.random.uniform(0.8, 2.2)
            total_time = num_cycles / frequency
            t = np.linspace(0, total_time, self.sequence_length)

            # 决定是否为纯净信号（60%概率）
            is_pure_signal = np.random.rand() < 0.3

            if signal_type == 0:  # 正弦波
                phase = np.random.uniform(0, 2 * np.pi)
                signal = amplitude * np.sin(2 * np.pi * frequency * t + phase)
                
                if not is_pure_signal:
                    # ADC量化模拟 - 8位ADC
                    quant_levels = 65536
                    # 将信号映射到0-amplitude范围进行量化
                    signal_shifted = (signal + amplitude) / 2  # 从[-amplitude, amplitude]映射到[0, amplitude]
                    signal_quantized = np.round(signal_shifted * (quant_levels - 1) / amplitude) 
                    signal = signal_quantized * amplitude / (quant_levels - 1) * 2 - amplitude  # 映射回原范围

                
            elif signal_type == 1:  # 方波 - 考虑对称和非对称两种情况
                # 随机选择方波类型：对称（关于0对称）或非对称（0到amplitude）
                symmetric_wave = np.random.choice([True, False])
                
                # 随机占空比，范围扩大到10%-90%
                duty_cycle = np.random.uniform(0.3, 0.7)
                
                if symmetric_wave:
                    # 对称方波：低电平为-amplitude，高电平为amplitude
                    # 随机相位决定起始电平
                    phase_choice = np.random.choice([0, 2*np.pi])
                    raw_square = amplitude * square(2 * np.pi * frequency * t + phase, duty=duty_cycle)
                    signal = raw_square  # 直接使用，范围是[-amplitude, amplitude]
                    
                    if not is_pure_signal:
                        # ADC量化模拟 - 需要将信号映射到ADC可接受的范围内
                        # 对于对称方波，我们假设ADC有双极性输入范围
                        signal_shifted = (signal + amplitude) / 2  # 映射到[0, amplitude]
                        quant_levels = 65536
                        signal_quantized = np.round(signal_shifted * (quant_levels - 1) / amplitude)
                        signal = signal_quantized * amplitude / (quant_levels - 1) * 2 - amplitude  # 映射回[-amplitude, amplitude]
                       
                else:
                    # 非对称方波：低电平为0，高电平为amplitude
                    # 随机相位决定起始电平
                    phase_choice = np.random.choice([0, 2*np.pi])
                    raw_square = 0.5 * amplitude * square(2 * np.pi * frequency * t + phase, duty=duty_cycle) + 0.5 * amplitude
                    signal = raw_square  # 范围是[0, amplitude]
                    
                    if not is_pure_signal:
                        # ADC量化模拟
                        quant_levels = 65536
                        signal_quantized = np.round(signal * (quant_levels - 1) / amplitude)
                        signal = signal_quantized * amplitude / (quant_levels - 1)          
                    # 根据方波类型进行最终裁剪
                    if symmetric_wave:
                        signal = np.clip(signal, -amplitude, amplitude)
                    else:
                        signal = np.clip(signal, 0, amplitude)
                
            elif signal_type == 2:  # 三角波/锯齿波
                # 随机相位
                phase = np.random.uniform(0, 2 * np.pi)
                
                # 随机选择对称或非对称
                symmetric_triangle = np.random.choice([True, False])
                
                # 锯齿波的对称性0-100%随机（50%为三角波）
                symmetry = np.random.uniform(0, 1)  # 0-100%对称性
                
                if symmetric_triangle:
                    # 对称三角波：在[-amplitude, amplitude]范围内
                    if symmetry <= 0.5:  # 三角波
                        signal = amplitude * sawtooth(2 * np.pi * frequency * t + phase, width=0.5)
                    else:  # 对称锯齿波
                        signal = amplitude * sawtooth(2 * np.pi * frequency * t + phase, width=symmetry)
                    
                    if not is_pure_signal:
                        # ADC量化模拟
                        signal_shifted = (signal + amplitude) / 2  # 映射到[0, amplitude]
                        quant_levels = 65536
                        signal_quantized = np.round(signal_shifted * (quant_levels - 1) / amplitude)
                        signal = signal_quantized * amplitude / (quant_levels - 1) * 2 - amplitude

                else:
                    # 非对称三角波：在[0, amplitude]范围内
                    if symmetry <= 0.5:  # 非对称三角波
                        triangle = sawtooth(2 * np.pi * frequency * t + phase, width=0.5)
                        signal = amplitude * (triangle + 1) / 2  # 映射到[0, amplitude]
                    else:  # 非对称锯齿波
                        saw = sawtooth(2 * np.pi * frequency * t + phase, width=symmetry)
                        signal = amplitude * (saw + 1) / 2  # 映射到[0, amplitude]
                    
                    if not is_pure_signal:
                        # ADC量化模拟
                        quant_levels = 65536
                        signal_quantized = np.round(signal * (quant_levels - 1) / amplitude)
                        signal = signal_quantized * amplitude / (quant_levels - 1)


            else:  # 其他信号类型
                signal = self._generate_other_signals(amplitude, frequency, phase, t)
                
                # 对于其他信号，也添加ADC量化模拟
                if not is_pure_signal:
                    # 将信号映射到合适的范围进行量化
                    signal_range = np.max(signal) - np.min(signal)
                    if signal_range > 0:
                        # 将信号归一化到[0, amplitude]范围
                        signal_normalized = (signal - np.min(signal)) / signal_range * amplitude
                        quant_levels = 65536
                        signal_quantized = np.round(signal_normalized * (quant_levels - 1) / amplitude)
                        signal = signal_quantized * amplitude / (quant_levels - 1)
                        # 映射回原范围
                        signal = signal / amplitude * signal_range + np.min(signal)
            
            base_noise = np.random.normal(0, amplitude * noise_level, len(signal))
            signal += base_noise
            # 将信号从电压范围(-5V到5V)映射到0-255
            # 假设信号范围是[-5V, 5V]映射到ADC 0-255
            voltage_range = 7.0  # -5V to +5V
            adc_offset = 32768      # 0V对应ADC值128
            adc_scale = 32767.5     # ±5V对应ADC值0-255
            
            # 将电压信号转换为ADC值
            adc_signal = (signal / voltage_range * 2) * adc_scale + adc_offset
            adc_signal = np.clip(adc_signal, 0, 65535)
            
            # 归一化到[0,1]
            signal_normalized = adc_signal / 65535.0
            
            self.signals.append(signal_normalized.astype(np.float32))
            self.labels.append(signal_type)
        
        self.signals = np.array(self.signals)
        self.labels = np.array(self.labels)
        
        # 统计信号类型分布
        unique, counts = np.unique(self.labels, return_counts=True)
        type_names = ['正弦波', '方波', '三角波', '其他']
        print("信号类型分布:")
        for type_id, count in zip(unique, counts):
            print(f"  {type_names[type_id]}: {count} 样本 ({count/len(self.labels)*100:.1f}%)")
        
        print(f"数据集生成完成: {len(self.signals)}个样本")
    
    def __len__(self):
        return len(self.signals)
    
    def __getitem__(self, idx):
        signal = torch.tensor(self.signals[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return signal, label

class SignalClassifier(nn.Module):
    """改进的MLP信号分类器模型"""
    def __init__(self, input_size=64, hidden_sizes=[32,16], num_classes=4, dropout_rate=0.05):
        super(SignalClassifier, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # 构建更深的隐藏层
        for i, hidden_size in enumerate(hidden_sizes):
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        
        self.feature_layers = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_size, num_classes)
        
        # 权重初始化
        self._initialize_weights()
    
    def _initialize_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        features = self.feature_layers(x)
        output = self.classifier(features)
        return output

class SignalTrainer:
    """信号分类器训练器"""
    def __init__(self, model, device, model_save_path='signal_classifier.pth', use_adaptive_loss=True):
        self.model = model.to(device)
        self.device = device
        self.model_save_path = model_save_path
        self.use_adaptive_loss = use_adaptive_loss
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        # 添加早停机制（patience增大到50）
        self.early_stopping = EarlyStopping(patience=50, min_delta=0.001, verbose=True)
        
    def train(self, train_loader, val_loader, num_epochs=150, learning_rate=0.001):
        """训练模型"""
        if self.use_adaptive_loss:
            # 20251108-修改权重没用
            base_weights = torch.tensor([
                [1.0, 1.0, 1.0, 1.0],  # 正弦波：增加与三角波和其他的混淆惩罚
                [1.0, 1.0, 1.5, 1.0],  # 方波：大幅提高与三角波的混淆惩罚（3.5）
                [1.5, 1.0, 1.0, 1.0],  # 三角波：提高与方波和正弦波的混淆惩罚
                [1.0, 1.5, 1.0, 1.0]   # 其他：保持合理
            ])
            criterion = AdaptiveFocusedLoss(num_classes=4, base_weights=base_weights, 
                                          confusion_penalty=0.5)  # 提高惩罚强度
            print("使用优化后的AdaptiveFocusedLoss进行训练")
        else:
            criterion = nn.CrossEntropyLoss()
            print("使用标准CrossEntropyLoss进行训练")
            
        # 优化：使用AdamW优化器，增加权重衰减
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=5e-4)  # 增加weight_decay
        
        # 改进：使用余弦退火调度器，周期更长
        #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs*2, eta_min=1e-6)
        # 在train()函数中
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=num_epochs,  # 改为num_epochs，避免过度衰减
            eta_min=1e-5       # 提高最低学习率，防止收敛过慢
        )
        print("开始训练...")
        best_val_accuracy = 0.0
        
        for epoch in range(num_epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for signals, labels in train_loader:
                signals, labels = signals.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(signals)
                loss = criterion(outputs, labels)
                loss.backward()
                
                # 梯度裁剪 - 保持
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            train_accuracy = 100 * train_correct / train_total
            avg_train_loss = train_loss / len(train_loader)
            
            # 验证阶段
            val_accuracy, avg_val_loss = self.validate(val_loader, criterion)
            
            # 记录损失和准确率
            self.train_losses.append(avg_train_loss)
            self.val_losses.append(avg_val_loss)
            self.train_accuracies.append(train_accuracy)
            self.val_accuracies.append(val_accuracy)
            
            # 更新学习率调度器
            scheduler.step()
            
            # 早停检查 - 使用验证损失
            self.early_stopping(avg_val_loss, self.model)
            
            # 保存最佳模型
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                self.save_model()
                print(f'保存最佳模型，验证准确率: {val_accuracy:.2f}%')
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}]')
                print(f'训练损失: {avg_train_loss:.4f}, 训练准确率: {train_accuracy:.2f}%')
                print(f'验证损失: {avg_val_loss:.4f}, 验证准确率: {val_accuracy:.2f}%')
                print(f'当前学习率: {optimizer.param_groups[0]["lr"]:.6f}')
                print('-' * 50)
            
            # 如果触发早停，恢复最佳模型并退出训练
            if self.early_stopping.early_stop:
                print("恢复早停前的最佳模型...")
                self.model.load_state_dict(self.early_stopping.best_model_state)
                break
        
        print(f"训练完成，最佳验证准确率: {best_val_accuracy:.2f}%")
    
    def validate(self, val_loader, criterion):
        """验证模型"""
        self.model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for signals, labels in val_loader:
                signals, labels = signals.to(self.device), labels.to(self.device)
                
                outputs = self.model(signals)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_accuracy = 100 * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        
        return val_accuracy, avg_val_loss
    
    def save_model(self):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_architecture': self.model.__class__.__name__,
            'input_size': 64,
            'num_classes': 4,
            'use_adaptive_loss': self.use_adaptive_loss
        }, self.model_save_path)
        print(f"模型已保存到: {self.model_save_path}")
    
    def plot_training_history(self):
        """绘制训练历史"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='训练损失')
        plt.plot(self.val_losses, label='验证损失')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('训练和验证损失')
        
        plt.subplot(1, 2, 2)
        plt.plot(self.train_accuracies, label='训练准确率')
        plt.plot(self.val_accuracies, label='验证准确率')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.title('训练和验证准确率')
        
        plt.tight_layout()
        plt.savefig('training_history_optimized.png', dpi=300, bbox_inches='tight')
        plt.show()

# 其余类保持不变
class SignalPredictor:
    """信号预测器"""
    def __init__(self, model_path, device):
        self.device = device
        
        checkpoint = torch.load(model_path, map_location=device)
        self.model = SignalClassifier(
            input_size=64,
            hidden_sizes=[32,16],
            num_classes=4,
            dropout_rate=0.05
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()
        
        self.class_names = ['正弦波', '方波', '三角波', '其他']
        print("模型加载完成 - 优化后的自适应损失训练")
    
    def predict(self, signal):
        """预测信号类型"""
        with torch.no_grad():
            signal_tensor = torch.tensor(signal, dtype=torch.float32).to(self.device)
            if signal_tensor.dim() == 1:
                signal_tensor = signal_tensor.unsqueeze(0)
            
            outputs = self.model(signal_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            max_prob, predicted = torch.max(probabilities, 1)
            
            final_predictions = [predicted[i].item() for i in range(len(predicted))]
            confidence_scores = [max_prob[i].item() for i in range(len(predicted))]
            
            return final_predictions, confidence_scores, probabilities.cpu().numpy()

def test_model(model, test_loader, device, class_names=['正弦波', '方波', '三角波', '其他']):
    """测试模型性能"""
    model.eval()
    all_predictions = []
    all_labels = []
    all_confidences = []
    
    with torch.no_grad():
        for signals, labels in test_loader:
            signals = signals.to(device)
            outputs = model(signals)
            probabilities = torch.softmax(outputs, dim=1)
            max_prob, predictions = torch.max(probabilities, 1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_confidences.extend(max_prob.cpu().numpy())
    
    accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))
    print(f"测试准确率: {accuracy * 100:.2f}%")
    
    from sklearn.metrics import classification_report, confusion_matrix
    print("\n分类报告:")
    print(classification_report(all_labels, all_predictions, 
                              target_names=class_names))
    
    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('混淆矩阵 - 优化后的自适应损失')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.tight_layout()
    plt.savefig('confusion_matrix_optimized.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    analyze_class_performance(all_predictions, all_labels, class_names)

def analyze_class_performance(predictions, labels, class_names):
    """分析各类别性能"""
    predictions = np.array(predictions)
    labels = np.array(labels)
    
    print("\n=== 各类别详细分析 ===")
    for i, class_name in enumerate(class_names):
        class_mask = labels == i
        class_total = np.sum(class_mask)
        
        if class_total > 0:
            class_accuracy = np.mean(predictions[class_mask] == labels[class_mask])
            
            wrong_predictions = predictions[class_mask] != labels[class_mask]
            if np.sum(wrong_predictions) > 0:
                wrong_classes, wrong_counts = np.unique(predictions[class_mask][wrong_predictions], return_counts=True)
                main_error = f"{class_names[wrong_classes[np.argmax(wrong_counts)]]}({wrong_counts[np.argmax(wrong_counts)]})"
            else:
                main_error = "无"
            
            print(f"{class_name}: 准确率={class_accuracy*100:.2f}%, 总样本={class_total}, 主要误分类={main_error}")

def main():
    """主函数"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 参数设置 - 建议20000:32:100增加
    num_samples = 80000
    batch_size = 128
    num_epochs = 400
    learning_rate = 0.001  # 降低学习率
    use_adaptive_loss = True  # 使用优化后的自适应损失
    
    # 创建数据集
    train_dataset = SignalDataset(num_samples=int(num_samples*0.9), include_other=True)
    test_dataset = SignalDataset(num_samples=int(num_samples*0.1), include_other=True)
    
    # 分割训练集
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"训练集: {len(train_dataset)} 样本")
    print(f"验证集: {len(val_dataset)} 样本")
    print(f"测试集: {len(test_dataset)} 样本")
    
    # 创建改进的模型
    model = SignalClassifier(
        input_size=64,
        hidden_sizes=[32, 16],  # 保持不变
        num_classes=4,
        dropout_rate=0.05
    )
    
    print(f"模型参数总数: {sum(p.numel() for p in model.parameters()):,}")
    
    # 创建训练器并开始训练
    trainer = SignalTrainer(model, device, 'best_signal_classifier_optimized.pth', 
                           use_adaptive_loss=use_adaptive_loss)
    trainer.train(train_loader, val_loader, num_epochs, learning_rate)
    
    # 绘制训练历史
    trainer.plot_training_history()
    
    # 加载最佳模型进行测试
    checkpoint = torch.load('best_signal_classifier_optimized.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 测试模型
    test_model(model, test_loader, device)
    
    # 保存最终的模型
    final_model_path = 'final_signal_classifier_optimized.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_architecture': model.__class__.__name__,
        'input_size': 64,
        'num_classes': 4,
        'class_names': ['正弦波', '方波', '三角波', '其他'],
        'use_adaptive_loss': use_adaptive_loss
    }, final_model_path)
    
    print(f"\n最终模型已保存到: {final_model_path}")
    
    # 创建预测器并演示
    predictor = SignalPredictor(final_model_path, device)
    
    # 演示预测
    demo_predictions(predictor, device)
    
    print("训练和测试完成!")

def demo_predictions(predictor, device):
    """演示预测功能"""
    print("\n=== 预测演示 ===")
    
    demo_dataset = SignalDataset(num_samples=10, include_other=True)
    
    for i in range(5):
        signal, true_label = demo_dataset[i]
        signal = signal.numpy()
        
        predictions, confidences, all_probs = predictor.predict(signal)
        
        true_class = predictor.class_names[true_label]
        pred_class = predictor.class_names[predictions[0]]
        confidence = confidences[0]
        
        print(f"信号 {i+1}:")
        print(f"  真实类别: {true_class}")
        print(f"  预测类别: {pred_class}")
        print(f"  置信度: {confidence:.4f}")
        print(f"  各类别概率: 正弦波={all_probs[0][0]:.4f}, 方波={all_probs[0][1]:.4f}, 三角波={all_probs[0][2]:.4f}, 其他={all_probs[0][3]:.4f}")
        print()

def visualize_sample_signals():
    """可视化一些样本信号"""
    dataset = SignalDataset(num_samples=12, include_other=True)
    class_names = ['正弦波', '方波', '三角波', '其他']
    
    plt.figure(figsize=(12, 9))
    for i in range(12):
        signal, label = dataset[i]
        plt.subplot(3, 4, i+1)
        plt.plot(signal.numpy())
        plt.title(f'{class_names[label]}')
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('sample_signals_optimized.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    visualize_sample_signals()
    main()