
import os
import pandas as pd

from data_processor import get_pred_result
from flask import Flask, render_template, redirect, url_for, request, flash, send_file, send_from_directory
from flask_uploads import UploadSet, configure_uploads, TEXT, patch_request_class
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import SubmitField
from werkzeug.utils import secure_filename


app = Flask(__name__, static_url_path='/static', static_folder='static')
app.config['SECRET_KEY'] = 'I have a dream'
app.config['UPLOADED_TRAINTEXTS_DEST'] = './data/train_data'  # 训练数据保存路径
app.config['UPLOADED_PREDTEXTS_DEST'] = './data/pred_data'  # 预测数据保存路径

predtexts = UploadSet('predtexts', TEXT)
configure_uploads(app, predtexts)
traintexts = UploadSet('traintexts', TEXT)
configure_uploads(app, traintexts)


class UploadForm(FlaskForm):
    """
    定义上传表单，包含一个文件字段和一个提交按钮
    """
    predtext = FileField(validators=[FileAllowed(predtexts, '请上传一个txt文件！'), FileRequired('选择预测文件上传')])
    traintext = FileField(validators=[FileAllowed(traintexts, '请上传一个txt文件！'), FileRequired('选择训练文件上传')])
    predsubmit = SubmitField('上传预测文件')
    trainsubmit = SubmitField('上传训练文件')
    run_modelsubmit = SubmitField('开始预测')


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    # print("upload_file")
    form = UploadForm()
    if form.is_submitted():
        if 'predsubmit' in request.form:
            # 处理预测文本文件上传
            for f in request.files.getlist('predtext'):
                original_filename = secure_filename(f.filename)
                filename = original_filename
                predtexts.save(f, name=filename)
        elif 'trainsubmit' in request.form:
            # 处理训练文本文件上传
            for f in request.files.getlist('traintext'):
                original_filename = secure_filename(f.filename)
                filename = original_filename
                traintexts.save(f, name=filename)
        elif 'run_modelsubmit' in request.form:
            # print("run_model")
            return redirect(url_for('run_model'))
        success = True
    else:
        success = False
    return render_template('index.html', form=form, success=success)


@app.route('/manage_pred_file')
def manage_pred_file():
    """
    处理文件管理的路由，列出所有上传的文件
    :return: 渲染的文件管理页面，携带文件列表
    """
    # 列出上传目录中的所有文件
    files_list = os.listdir(app.config['UPLOADED_PREDTEXTS_DEST'])
    return render_template('pred_file_manage.html', files_list=files_list)


@app.route('/download_pred_file/<filename>')
def download_pred_file(filename):
    """
    处理文件下载的路由，将指定的文件下载到客户端
    :param filename: 要下载的文件名
    :return: 指定文件的二进制数据
    """
    # 获取文件的路径并下载
    file_path = predtexts.path(filename)
    new_path = file_path.replace('\\', '/')
    new_path = new_path.replace('./', '')
    return send_file(new_path, as_attachment=True)
    # return redirect(url_for('manage_pred_file'))


@app.route('/delete_pred_file/<filename>')
def delete_pred_file(filename):
    """
    处理文件删除的路由，从服务器上移除指定的文件
    :param filename: 要删除的文件名
    :return: 重定向到文件管理页面
    """
    # 获取文件的路径并删除
    file_path = predtexts.path(filename)
    os.remove(file_path)
    return redirect(url_for('manage_pred_file'))


# 新增一个管理训练数据的路由
@app.route('/manage_train_file')
def manage_train_file():
    """
    处理训练文件管理的路由，列出所有上传的训练文件
    :return: 渲染的训练文件管理页面，携带文件列表
    """
    # 列出训练数据目录中的所有文件
    files_list = os.listdir(app.config['UPLOADED_TRAINTEXTS_DEST'])
    return render_template('train_file_manage.html', files_list=files_list)


@app.route('/download_train_file/<filename>')
def download_train_file(filename):
    """
    处理文件下载的路由，将指定的文件下载到客户端
    :param filename: 要下载的文件名
    :return: 指定文件的二进制数据
    """
    # 获取文件的路径并下载
    file_path = traintexts.path(filename)
    new_path = file_path.replace('\\', '/')
    new_path = new_path.replace('./', '')
    return send_file(new_path, as_attachment=True)


# 更新删除文件的路由，使其可以删除训练文件
@app.route('/delete_train_file/<filename>')
def delete_train_file(filename):
    """
    处理训练文件删除的路由，从服务器上移除指定的训练文件
    :param filename: 要删除的训练文件名
    :return: 重定向到训练文件管理页面
    """
    # 获取训练文件的路径并删除
    file_path = traintexts.path(filename)
    os.remove(file_path)
    return redirect(url_for('manage_train_file'))


@app.route('/run_model')
def run_model():
    # print("调用run_model")
    # 检查是否有预测文件上传
    pred_files = os.listdir(app.config['UPLOADED_PREDTEXTS_DEST'])
    if not pred_files:
        flash('请先上传预测文件')
        return redirect(url_for('upload_file'))

    # 获取预测结果，这里只是一个示例，你需要根据实际情况调用get_pred_result
    score, mse, res = get_pred_result()
    res_html = res.to_html(justify='center', index=False)  # 将DataFrame转换为HTML表格字符串，index=False表示不包含索引列

    return render_template('report.html', res_html=res_html, score=score, mse=mse)


@app.route('/download_report')
def download_report():
    file_path = r'data/report/report.xlsx'
    directory_path = r'data/report'

    if os.path.isfile(file_path):
        # return send_from_directory(directory=directory_path, path=file_path, as_attachment=True)
        return send_file(file_path, as_attachment=True)
    else:
        flash('请先运行模型进行预测，生成报告')
        return redirect(url_for('upload_file'))


if __name__ == '__main__':
    app.run()