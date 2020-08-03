import os
root_dir = 'data/kitti_example'
directories = []
lengths = []
for folder in os.listdir(root_dir):
    if folder.endswith("zip") == False:
        new_dir = os.path.join(root_dir, folder)
        for folder2 in os.listdir(new_dir):
            if folder2.endswith("txt") == False:
                new_dir2 = os.path.join(new_dir, folder2)
                directories.append(os.path.join(folder, folder2))
                for folder3 in os.listdir(new_dir2):
                    new_dir3 = os.path.join(new_dir2, folder3)
                    for folder4 in os.listdir(new_dir3):
                        new_dir4 = os.path.join(new_dir3, folder4)
                        count = 0
                        for folder5 in os.listdir(new_dir4):
                            count+=1
                        lengths.append(count)
                        break
                    break
                break
            break
for i, direc in enumerate(directories):
    print(direc + " " + str(lengths[i]))