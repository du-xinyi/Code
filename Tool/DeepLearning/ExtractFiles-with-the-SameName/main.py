import shutil
import os

def main():
	# 提取名称的目标
	path_label = './val/labels'
	# 需要从里面挑出来同名的文件
	path_object = './val/images'
	type_object = 'png'
	# 输出路径
	path_output = './images/'

	if not os.path.exists(path_output):
		os.makedirs(path_output)
	for i in os.walk(path_label):
		for j in i[2]:
			p_label = os.path.join(path_label, j)
			# 默认label内是txt文件，长度为3
			obj_name = j[:-3]+type_object
			print(obj_name)
			# 挑选出来的同名文件路径
			obj_path = os.path.join(path_object, obj_name)
			if os.path.exists(obj_path) == True:
				new_path = os.path.join(path_output, obj_name)
				# shutil.copyfile(obj_path,new_path) # 拷贝
				shutil.move(obj_path, new_path) # 移动

if __name__ == '__main__':
	main()