# -这里是具体的坐标转换代码，里面由两部分需要根据自己的数据进行调整。第一部分是坐标数据，需要输入自己的已知四个像素点和对应的实际坐标点（像素点坐标可通过imagej读取获得）。一二部分是坐标对应的图片的地址和要输出到的Excel表格的地址，根据自己的存储地址来输入即可。#
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import os
from scipy.optimize import least_squares

import PIL.Image
PIL.Image.MAX_IMAGE_PIXELS = None  

class InteractiveCoordinateConverter:
    def __init__(self, image_path, output_path=None):
        """
        
        Args:
            image_path: 图片文件路径
            output_path: 输出Excel文件路径
        """
        self.image_path = image_path
        self.output_path = output_path or r"Excel表格的地址"
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        
        # 设置中文字体
        self.setup_chinese_font()
        
        # 初始化数据
        self.control_points_2d = []
        self.control_points_3d = []
        self.marked_points_2d = []
        self.marked_points_3d = []
        self.transform_matrix = None
        self.is_calibrated = False
        
        # 用于存储图形对象
        self.marked_circles = []
        self.marked_texts = []
        
        # 缩放相关
        self.zoom_factor = 1.0
        self.zoom_step = 0.2
        self.current_center = None
        
        # 工具栏状态跟踪
        self.is_pan_mode = False
        
        # 加载图片
        self.load_image()
        
        # 设置控制点
        self.setup_control_points()
        
        print("=" * 60)
        print("交互式坐标转换工具")
        print("=" * 60)
        print("请在图片上点击标记点")
        print("关闭图片窗口后自动导出Excel")
        print("=" * 60)
    
    def setup_chinese_font(self):
        """设置中文字体支持"""
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 支持中文的字体
        plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
    
    def load_image(self):
        """加载图片"""
        import matplotlib.image as mpimg
        print(f"正在加载图片: {os.path.basename(self.image_path)}")
        print("图片尺寸较大，加载可能需要一些时间...")
        
        self.image = mpimg.imread(self.image_path)
        self.image_height, self.image_width = self.image.shape[:2]
        
        print(f"图片已加载: {os.path.basename(self.image_path)}")
        print(f"图片尺寸: {self.image_width} x {self.image_height} 像素")
        print(f"总像素数: {self.image_width * self.image_height:,}")
    
    def setup_control_points(self):
        """设置控制点"""
        # 控制点1: 左上角（修改）
        self.control_points_2d.append([xxx, xxx])
        self.control_points_3d.append([xxxxxxx, xxxxxx, xxx])
        
        # 控制点2: 右上角（修改）
        self.control_points_2d.append([xxxx, xxxx])
        self.control_points_3d.append([xxxxxxx, xxxxxx, xxx])
        
        # 控制点3: 左下角（修改）（一般可设hhh1=hhh2）
        self.control_points_2d.append([xxxx, xxxx])
        self.control_points_3d.append([xxxxxxx, xxxxxx, hhh1])
        
        # 控制点4: 右下角（修改）
        self.control_points_2d.append([xxxx, xxxx])
        self.control_points_3d.append([xxxxxxx, xxxxxx, hhh2])
        
        print("控制点已设置:")
        for i, (pixel, real) in enumerate(zip(self.control_points_2d, self.control_points_3d)):
            print(f"  控制点{i+1}: 像素{pixel} -> 实际{real}")
    
    def calculate_transformation(self):
        """计算转换关系"""
        if len(self.control_points_2d) < 3:
            print("错误: 至少需要3个控制点")
            return False
        
        print("计算转换关系...")
        
        # 使用最小二乘法计算变换矩阵
        def residual_func(params, points_2d, points_3d):
            matrix = params.reshape(3, 3)
            residuals = []
            
            for i in range(len(points_2d)):
                x, y = points_2d[i]
                predicted = matrix @ np.array([x, y, 1])
                residual = predicted - points_3d[i]
                residuals.extend(residual)
            
            return np.array(residuals)
        
        initial_guess = np.eye(3).flatten()
        result = least_squares(residual_func, initial_guess, 
                              args=(self.control_points_2d, self.control_points_3d), method='lm')
        
        self.transform_matrix = result.x.reshape(3, 3)
        
        # 验证精度
        errors = []
        for i in range(len(self.control_points_2d)):
            calculated = self.pixel_to_3d(self.control_points_2d[i][0], self.control_points_2d[i][1])
            expected = self.control_points_3d[i]
            error = np.linalg.norm(calculated - expected)
            errors.append(error)
        
        avg_error = np.mean(errors)
        max_error = np.max(errors)
        
        print(f"转换精度验证:")
        print(f"  平均误差: {avg_error:.6f} 单位")
        print(f"  最大误差: {max_error:.6f} 单位")
        
        if avg_error > 1.0:
            print("警告: 转换精度较低，请检查控制点坐标")
        else:
            print("转换关系建立成功!")
        
        self.is_calibrated = True
        return True
    
    def pixel_to_3d(self, x, y):
        """将像素坐标转换为实际坐标"""
        if self.transform_matrix is None:
            return None
        return self.transform_matrix @ np.array([x, y, 1.0])
    
    def start_interactive_marking(self):
        """开始交互式标记"""
        if not self.is_calibrated:
            if not self.calculate_transformation():
                return
        
        # 创建图形窗口
        self.fig, self.ax = plt.subplots(figsize=(15, 10))
        
        # 启用交互式导航工具栏（包含缩放、平移等功能）
        self.fig.canvas.manager.set_window_title('Interactive Coordinate Marker')
        
        # 显示图片
        self.ax.imshow(self.image)
        self.ax.set_title('Click on image to mark points (Red=Control, Blue=Marked)\nClose window to export Excel\nUse toolbar or A/Z keys to zoom')
        
        # 显示控制点
        for i, (x, y) in enumerate(self.control_points_2d):
            circle = Circle((x, y), radius=20, color='red', fill=False, linewidth=2)
            self.ax.add_patch(circle)
            self.ax.text(x + 25, y - 25, f'C{i+1}', color='red', fontsize=12, weight='bold')
        
        # 显示已有标记点
        self.update_marked_points_display()
        
        # 设置点击事件
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        
        # 添加说明文本
        self.status_text = self.ax.text(10, 30, f"Marked {len(self.marked_points_2d)} points", 
                                       color='white', fontsize=12, 
                                       bbox=dict(facecolor='black', alpha=0.7))
        
        # 添加键盘事件
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        
        # 添加鼠标移动事件，用于跟踪鼠标位置
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        
        # 添加操作说明
        instructions = (
            "Instructions:\n"
            "- Click image to add points\n"
            "- Use toolbar to zoom/pan\n"
            "- 'A': zoom in at mouse position\n"
            "- 'Z': zoom out at mouse position\n"
            "- 'd': delete last point\n"
            "- 'r': reset all points\n"
            "- 'e': export points\n"
            "- 'h': home view\n"
            "- 'p': toggle pan mode\n"
            "- Close window to export"
        )
        self.ax.text(self.image_width - 300, 30, instructions, 
                    color='white', fontsize=10, 
                    bbox=dict(facecolor='black', alpha=0.7),
                    verticalalignment='top')
        
        print("开始交互式标记:")
        print("- 在图片上点击添加标记点")
        print("- 使用工具栏缩放和平移图片")
        print("- 按 'A' 放大鼠标位置")
        print("- 按 'Z' 缩小鼠标位置")
        print("- 按 'd' 删除最后一个标记点")
        print("- 按 'r' 重置所有标记点")
        print("- 按 'e' 导出并继续标记")
        print("- 按 'h' 回到完整视图")
        print("- 按 'p' 切换平移模式")
        print("- 关闭窗口自动导出")
        
        # 使用block=True确保窗口保持打开状态
        plt.tight_layout()
        plt.show(block=True)
        
        # 窗口关闭后自动导出
        self.export_to_excel()
    
    def on_click(self, event):
        """处理点击事件"""
        # 如果在平移模式下，不添加标记点
        if self.is_pan_mode:
            return
            
        if event.inaxes != self.ax:
            return
            
        if event.xdata is not None and event.ydata is not None:
            x, y = int(event.xdata), int(event.ydata)
            
            # 转换为实际坐标
            actual_coords = self.pixel_to_3d(x, y)
            
            if actual_coords is not None:
                # 添加到标记点列表
                self.marked_points_2d.append([x, y])
                self.marked_points_3d.append(actual_coords)
                
                # 更新显示
                self.update_marked_points_display()
                
                print(f"标记点 {len(self.marked_points_2d)}: 像素({x}, {y}) -> 实际({actual_coords[0]:.2f}, {actual_coords[1]:.2f}, {actual_coords[2]:.2f})")
    
    def on_mouse_move(self, event):
        """处理鼠标移动事件，记录当前鼠标位置"""
        if event.inaxes == self.ax and event.xdata is not None and event.ydata is not None:
            self.current_center = (event.xdata, event.ydata)
    
    def zoom_at_point(self, zoom_in=True):
        """以当前鼠标位置为中心进行缩放
        
        Args:
            zoom_in: True表示放大，False表示缩小
        """
        if self.current_center is None:
            # 如果没有鼠标位置，使用视图中心
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()
            self.current_center = ((xlim[0] + xlim[1]) / 2, (ylim[0] + ylim[1]) / 2)
        
        x, y = self.current_center
        
        # 获取当前视图范围
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        
        # 计算缩放因子
        if zoom_in:
            # 放大 - 缩小视图范围
            factor = 1 / (1 + self.zoom_step)
        else:
            # 缩小 - 扩大视图范围
            factor = 1 + self.zoom_step
        
        # 计算新的视图范围
        new_width = (xlim[1] - xlim[0]) * factor
        new_height = (ylim[1] - ylim[0]) * factor
        
        # 以鼠标位置为中心设置新范围
        new_xlim = [x - new_width / 2, x + new_width / 2]
        new_ylim = [y - new_height / 2, y + new_height / 2]
        
        # 设置新范围
        self.ax.set_xlim(new_xlim)
        self.ax.set_ylim(new_ylim)
        
        # 更新显示
        self.fig.canvas.draw_idle()
        
        # 更新缩放因子
        if zoom_in:
            self.zoom_factor *= (1 + self.zoom_step)
        else:
            self.zoom_factor /= (1 + self.zoom_step)
            
        print(f"缩放级别: {self.zoom_factor:.2f}x ({'放大' if zoom_in else '缩小'})")
    
    def reset_view(self):
        """重置视图到完整图片"""
        self.ax.set_xlim(0, self.image_width)
        self.ax.set_ylim(self.image_height, 0)  # 注意：y轴是反向的
        self.zoom_factor = 1.0
        self.fig.canvas.draw_idle()
        print("视图已重置")
    
    def on_key(self, event):
        """处理键盘事件"""
        if event.key == 'd':  # 删除最后一个点
            if self.marked_points_2d:
                removed_point = self.marked_points_2d.pop()
                self.marked_points_3d.pop()
                print(f"已删除标记点: 像素{removed_point}")
                self.update_marked_points_display()
        
        elif event.key == 'r':  # 重置所有标记点
            self.marked_points_2d = []
            self.marked_points_3d = []
            print("已重置所有标记点")
            self.update_marked_points_display()
        
        elif event.key == 'e':  # 导出
            self.export_to_excel()
        
        elif event.key == 'h':  # 回到完整视图
            self.reset_view()
        
        elif event.key == 'p':  # 切换平移模式
            self.is_pan_mode = not self.is_pan_mode
            mode = "平移模式" if self.is_pan_mode else "标记模式"
            print(f"已切换到{mode}")
            self.update_marked_points_display()
        
        # 使用A键放大
        elif event.key == 'a':
            self.zoom_at_point(zoom_in=True)
        
        # 使用Z键缩小
        elif event.key == 'z':
            self.zoom_at_point(zoom_in=False)
    
    def update_marked_points_display(self):
        """更新标记点显示 - 优化版本，不在图片上显示坐标文本"""
        # 清除之前标记的图形对象
        for circle in self.marked_circles:
            circle.remove()
        for text in self.marked_texts:
            text.remove()
        self.marked_circles = []
        self.marked_texts = []
        
        # 显示标记点 - 蓝色
        for i, (x, y) in enumerate(self.marked_points_2d):
            circle = Circle((x, y), radius=15, color='blue', fill=False, linewidth=2)
            self.ax.add_patch(circle)
            self.marked_circles.append(circle)
            
            text_id = self.ax.text(x + 20, y + 20, f'M{i+1}', color='blue', fontsize=10)
            self.marked_texts.append(text_id)
        
        # 更新状态文本
        if hasattr(self, 'status_text'):
            mode_text = " (Pan Mode)" if self.is_pan_mode else ""
            self.status_text.set_text(f"Marked {len(self.marked_points_2d)} points{mode_text}")
        
        # 刷新显示
        self.fig.canvas.draw_idle()
    
    def export_to_excel(self):
        """导出到Excel"""
        if not self.marked_points_2d:
            print("没有标记点可导出")
            return
        
        # 创建DataFrame
        df = pd.DataFrame({
            '点编号': range(1, len(self.marked_points_2d) + 1),
            '像素X': [p[0] for p in self.marked_points_2d],
            '像素Y': [p[1] for p in self.marked_points_2d],
            'X坐标': [p[0] for p in self.marked_points_3d],
            'Y坐标': [p[1] for p in self.marked_points_3d],
            'Z坐标': [p[2] for p in self.marked_points_3d]
        })
        
        # 保存到Excel
        df.to_excel(self.output_path, index=False)
        
        print(f"\n已导出 {len(self.marked_points_2d)} 个点的坐标到:")
        print(f"  {self.output_path}")
        
        # 显示数据预览
        print("\n数据预览:")
        print(df.head())
        
        return df

def main():
    # 设置图片路径和输出路径
    image_path = r"图片路径"  # 使用原始字符串
    output_path = r"Excel表格路径"
    
    # 创建转换器
    converter = InteractiveCoordinateConverter(image_path, output_path)
    
    # 开始交互式标记
    converter.start_interactive_marking()

if __name__ == "__main__":
    main()
