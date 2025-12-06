# 红外退化： 

low contrast 

noise

strip noise 




# 可见光退化

blur 

haze 

noise 

rain 



# 数据集内容总结： 
## MSRS
	其中的ir 内容是使用cv2读取的是时候是三条相同的通道内容，图像的维度是 480 640 高质量的红外可见光图像内容； 
	这里的原因是默认的形式是img = cv2.imread(path)   # 等价于 cv2.IMREAD_COLOR
	使用cv2 imread 读取到的内容自动转换为了 numpy.ndarray 这样的形式； 

	但是要是选择使用PIL.Image 读取的到的就是 480 640 HW 这样图像了； np.array() 转换回去； 
	
	

# 个人的一点总结和理解内容： 

给图像加上去的噪声一般都是在图像的格式为  HWC （红外可以扩展为HW1这样的） np.ndarry uint8 的格式； 



# 一类：通用测试 / 验证 Notebook 名

**1. `test_and_validation.ipynb`**  
（直接表达：用于测试与验证）

**2. `debug_experiments.ipynb`**  
（用于临时实验、调试）

**3. `experiment_sandbox.ipynb`**  
（沙盒式 notebook，用于随手测试）

**4. `analysis_and_visualization.ipynb`**  
（如果主要是看输出、画图）

---

# 二类：针对 IVIF 或退化模拟的 Notebook 名

**5. `IVIF_Degradation_Test.ipynb`**  
（测试退化模拟模块）

**6. `IVIF_Model_Validation.ipynb`**  
（用于验证融合模型/模块）

**7. `IVIF_UnitTests.ipynb`**  
（做模块级的单元测试）

**8. `Degradation_Module_Demo.ipynb`**  
（展示与验证你的退化模块）

---

# 三类：用于方法开发 / 算法验证

**9. `Prototype_Testbench.ipynb`**  
（非常适合科研代码原型验证）

**10. `Method_Validation.ipynb`**  
（如果你要对新idea做验证）

**11. `FusionPipeline_Debug.ipynb`**  
（对你的 AE → Diffusion → FCM 流水线做局部验证）

---

# 四类：更短 / 更工程化的

**12. `dev_test.ipynb`**  
**13. `playground.ipynb`**  
**14. `scratch.ipynb`**

非常常见且轻量，用法灵活。