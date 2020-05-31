var express = require('express');
var router = express.Router();
var fs = require('fs');
var childProcess = require('child_process');
var multer  = require('multer');

var storage = multer.diskStorage({
  destination: function (req, file, cb) {
      cb(null, './public/asset/matrix');    // 保存的路径，备注：需要自己创建
  },
  filename: function (req, file, cb) {
      cb(null, file.originalname);  
  }
});

var upload = multer({ storage: storage })

/* GET home page. */
router.get('/', function(req, res, next) {
  res.render('algorithm');
});

router.get('/get_matrix_file',function(req,res,next){
  var filelist = fs.readdirSync('./public/asset/matrix');
  var exepath = './';
  var __exelist = fs.readdirSync(exepath);
  var exes = [];
  for(var i=0;i<__exelist.length;i++){
    if(__exelist[i].split('.')[1]==='exe'){
      exes.push(__exelist[i].split('.')[0])
    }
    else{
      continue
    }
  }
  res.send([filelist,exes]);
})

router.get('/execute_exe',function(req,res,next){
  var params = req.query;
  var exes = params.exenames;
  var info = {};
  for(var i=0;i<exes.length;i++){
    var temp ={};
    var exename = exes[i] + '.exe';
    var stdout = '';
    if(exename === "SOR.exe"){
      stdout = childProcess.execSync(exename+' '+params.dimension+' ./public/asset/matrix/'+params.matrixfile+' '
                   +params.precision+' ./public/asset/result/'+exes[i]+'_'+params.outputname+" "+params.sorparams);
    }
    else{
      stdout = childProcess.execSync(exename+' '+params.dimension+' ./public/asset/matrix/'+params.matrixfile+' '
                   +params.precision+' ./public/asset/result/'+exes[i]+'_'+params.outputname);
    }
    temp.msg = stdout.toString();
    var data = stdout.toString().split('\n')[0];
    data = data.split(' ');
    for(var j=0;j<data.length;j++){
      data[j] = parseInt(data[j]);
    }
    temp.data = data;
    info[exes[i]] = temp;
  }
  info.code = 1;
  res.send(info)
})

router.get('/get_result_txt',function(req,res,next){
  var params = req.query;
  var start = (parseInt(params.page) - 1)*parseInt(params.limit);
  var end = start + parseInt(params.limit);
  var filelist = fs.readdirSync('./public/asset/result');
  var fileinfo = [];
  for(var i=0;i<filelist.length;i++){
    var temp = {};
    var fstat = fs.statSync('./public/asset/result/'+filelist[i]);
    temp.ID = i+1;
    temp.FileName = filelist[i];
    temp.Modificationdate = fstat.mtime;
    temp.Type = 'txt';
    temp.FileSize = (Math.ceil(fstat.size/1024)).toString() + 'KB';
    fileinfo.push(temp);
  }
  res.send({code:0,msg:'',count:fileinfo.length,data:fileinfo.slice(start,end)})
})

router.post('/upload',upload.array('file', 20),function(req,res,next){
  res.json({
    "code": 0
    ,"msg": ""
    ,"data": {
    }
  })
})

router.get('/delete_file',function(req,res,next){
  var params = req.query;
  fs.unlink("./public/asset/result/"+params.FileName,function(err){
    if(err){
      res.send({code:0,msg:'删除失败!'})
      console.log(err)
    }
    else{
      res.send({code:1,msg:'删除成功!'})
    }
  })
})

module.exports = router;
