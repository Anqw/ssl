cd datasets
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
tar xvf VOCtrainval_06-Nov-2007.tar && rm VOCtrainval_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
tar xvf VOCtest_06-Nov-2007.tar && rm VOCtest_06-Nov-2007.tar
wget host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
tar xvf VOCtrainval_11-May-2012.tar && rm VOCtrainval_11-May-2012.tar
cd VOCdevkit
mv * ../
cd -
rm -rf VOCdevkit
cd -
