import os
import shutil
import sys

def clean_results_folder():
    """清空results文件夹中的所有文件"""
    results_dir = 'results'
    
    # 检查目录是否存在
    if not os.path.exists(results_dir):
        print("results目录不存在，无需清理。")
        return
    
    try:
        # 获取文件列表
        files = [f for f in os.listdir(results_dir) if os.path.isfile(os.path.join(results_dir, f))]
        
        if not files:
            print("results目录已经是空的。")
            return
        
        # 删除文件
        for file in files:
            file_path = os.path.join(results_dir, file)
            try:
                os.remove(file_path)
                print(f"已删除: {file}")
            except Exception as e:
                print(f"无法删除文件 {file_path}: {e}")
        
        print(f"已成功清空results目录，删除了 {len(files)} 个文件。")
        
    except Exception as e:
        print(f"清理过程中发生错误：{str(e)}")

if __name__ == '__main__':

    clean_results_folder()
