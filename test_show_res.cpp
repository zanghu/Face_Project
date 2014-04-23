#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <stdexcept>

using namespace std;
using namespace cv;

class Result {
    vector<string> name_vec; //每一幅图片的名字
    //vector<Mat> image_vec; //Mat格式的每一幅图
    vector<string> data_path; //每幅图的存储路径(含名字)
    vector< vector<Point2d> > kp_vec; //key points
    string dump_dir; //导出加入关键点的图片的目录，必须以/结尾 

public:
    //pic_path: 图片完整位置信息txt文件，包含文件名
    //name_path: 图片名txt，保留前一级目录lfw或net的信息，用于导出标识图
    //label_path: 图片关键点信息txt文件
    Result(const string& pic_path, const string& name_path, const string& label_path) {
        getDataPath(pic_path);
        getLabel(label_path);
        getName(name_path);
        if ((name_vec.size()!=data_path.size()) || (data_path.size()!=kp_vec.size())) {
            throw runtime_error("vectors not aligned\n");
        }
    }
    //path所在的文本，每一行是一个路径
    void getDataPath(const string& path) {
        ifstream in(path.c_str());
        if (!in) {
            cout << "in getPath, path error" << endl;
            return;
        }
        string cur_path;
        int cnt = 0;
        while (1) {
            getline(in, cur_path);
            if (in.eof()) break;
            this->data_path.push_back(cur_path);
            ++cnt;
        }
        cout << "in getDataPath, cnt=" << cnt << endl;
        in.close();
    }

    void getLabel(const string& path) {
        ifstream in(path.c_str());
        if (!in) {
            cout << "in getLable, error" << endl;
            return;
        }
        string line;
        int cnt = 0;
        while (1) {
            getline(in, line);
            if (in.eof()) break;
            istringstream iss(line);
            vector<cv::Point2d> temp_vec;
            string x;
            string y;
            int num = 5;
            while (num > 0) {
                iss >> x;
                iss >> y;
                temp_vec.push_back(Point2d(atof(x.c_str()), atof(y.c_str())));
                --num;
            }
            this->kp_vec.push_back(temp_vec);
        }
        cout << "in getLable, cnt=" << cnt << endl;
        in.close();
    }

    void getName(const string& path) {
        ifstream in(path.c_str());
        if (!in) {
            cout << "in getName, error" << endl;
            return;
        }
        string name;
        int cnt = 0;
        while (1) {
            getline(in, name);
            if (in.eof()) break;
            this->name_vec.push_back(name);
            ++cnt;
        }
        cout << "in getName, cnt=" << cnt << endl;
        in.close();
    }


    void generateMarkedImage(const string& dump_dir) const {
        for (int i = 0; i < this->data_path.size(); ++i) {
            //读取图片
            Mat image = imread(data_path[i], CV_LOAD_IMAGE_COLOR);
            //读取图片对应的所有关键点
            vector<cv::Point2d> temp_vec = kp_vec[i];
            //画出关键点
            for (int j = 0; j < 5; ++j) {
                circle(image, temp_vec[j], 1, Scalar(0, 255, 0), -1);
            }
            string dump_path(dump_dir + this->name_vec[i]);
            bool b = imwrite(dump_path, image);
            
            if (b==false) {
                cout << "path: " << dump_path << endl;
                cout << "imwrite false" << endl;
                return;
            }
            //break;
        }
        cout << "dump finish" << endl;
    }
    
    void generateMarkedImageWithDifferentColor(const string& dump_dir, const string& failure_txt) const {
        ifstream in(failure_txt.c_str());
        if (!in) {
            cout << "in generateMarkedImageWithDifferentColor, path error" << endl;
            return;
        }
        string line;
        for (int i = 0; i < this->data_path.size(); ++i) {
            //读取图片
            getline(in, line); //读取记录每个样本的模型输出关键点是否失败的txt文件的一行
            istringstream iss(line); //创建一个istringstream对象
            Mat image = imread(data_path[i], CV_LOAD_IMAGE_COLOR);
            //读取图片对应的所有关键点
            vector<cv::Point2d> temp_vec = kp_vec[i];
            //画出关键点
            for (int j = 0; j < 5; ++j) {
                string temp;
                iss >> temp;
                int tf_mark = atoi(temp.c_str());
                if (tf_mark == 0) circle(image, temp_vec[j], 1, Scalar(0, 255, 0), -1);
                else circle(image, temp_vec[j], 1, Scalar(255, 0, 0), -1);
            }
            string dump_path(dump_dir + this->name_vec[i]);
            bool b = imwrite(dump_path, image);
            
            if (b==false) {
                cout << "path: " << dump_path << endl;
                cout << "imwrite false" << endl;
                return;
            }
            //break;
        }
        in.close(); //关闭文件流
        cout << "dump finish" << endl;
    }

};

void dump_train(const string& pic_path, const string& name_path, const string& label_path, const string& dump_dir, const string& failure_txt) {

    Result res(pic_path, name_path, label_path);
    cout << "初始化完成" << endl;
    res.generateMarkedImageWithDifferentColor(dump_dir, failure_txt);
    cout << "图片生成完成" << endl;
}

void dump_test(const string& pic_path, const string& name_path, const string& label_path, const string& dump_dir, const string& failure_txt) {

    Result res(pic_path, name_path, label_path);
    cout << "初始化完成" << endl;
    res.generateMarkedImageWithDifferentColor(dump_dir, failure_txt);
    cout << "图片生成完成" << endl;
}

int main(int argc, char* argv[]) {
    //图片绝对路径
    cout <<"argc=" << argc << endl;
    cout << "argv[1]: " << argv[1] << endl;
    string pic_path(argv[1]); //"/home/zanghu/Pro_Datasets/yisun/train/roi/my_pic_trainImageList_roi");
    //图片名和上一级目录
    string name_path(argv[2]); //"/home/zanghu/Pro_Datasets/yisun/train/roi/my_train_nameList");
    //关键点坐标txt
    string label_path(argv[3]); //"/home/zanghu/Pro_Datasets/yisun/train/show_res/train_label.txt");
    //导出路径
    string dump_dir(argv[4]); //"/home/zanghu/Pro_Datasets/yisun/train/marked_iamges/train/"); //注意dump_dir必须以/结尾
    
    string failure_txt_train(argv[9]);

    dump_train(pic_path, name_path, label_path, dump_dir, failure_txt_train);
    
    
    //图片绝对路径
    string pic_path_test(argv[5]); //"/home/zanghu/Pro_Datasets/yisun/train/roi/my_pic_testImageList_roi");
    //图片名和上一级目录
    string name_path_test(argv[6]); //"/home/zanghu/Pro_Datasets/yisun/train/roi/my_test_nameList");
    //关键点坐标txt
    string label_path_test(argv[7]); //"/home/zanghu/Pro_Datasets/yisun/train/show_res/test_label.txt");
    
    string dump_dir_test(argv[8]); //"/home/zanghu/Pro_Datasets/yisun/train/marked_iamges/test/"); //注意dump_dir必须以/结尾
    
    string failure_txt_test(argv[10]);
    
    dump_test(pic_path_test, name_path_test, label_path_test, dump_dir_test, failure_txt_test);

    return 1;
}

