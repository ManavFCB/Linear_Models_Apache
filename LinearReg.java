package cs6240;
import java.io.*;
import java.util.*;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.*;
import java.net.URI;
import org.apache.hadoop.fs.*;
import org.apache.hadoop.mapreduce.*;
import org.apache.hadoop.mapreduce.lib.input.*;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.apache.log4j.*;
public class LinearReg extends Configured implements Tool {
    private static final Logger logger = LogManager.getLogger(LinearReg.class);
    private static long flag;
    private static double error;
    private static int num_it;
    private static double sum_error;
    enum Stats{
        curr_mse,flag;
    }
    public static class Train_mapper extends Mapper<Object, Text, IntWritable, Text> {
        private Map<Integer,Double> weight_up_sums;
        private Map<Integer,Double> weights=new HashMap<>();
        int nob=0;
        double sum_error=0.0;
        protected void setup(Context context) throws IOException, InterruptedException {
            URI[] cacheFiles = context.getCacheFiles();
            int noi=Integer.parseInt(context.getConfiguration().get("noi"));
            if (cacheFiles != null && cacheFiles.length > 0) {
                try {
                    logger.info(cacheFiles[0]);
                    File f1 = new File("./"+"output"+noi+".txt");
                    BufferedReader br = new BufferedReader(new FileReader(f1));
                    String l = br.readLine();
                        String[] s = l.split(",");
                    for(int i=1;i<s.length;i++){
                        weights.put(i-1,Double.parseDouble(s[i]));
                    }
                    weight_up_sums=new HashMap<>();
                    for(int i=0;i<weights.size();i++)
                        weight_up_sums.put(i,0.0);
                } catch (Exception e) {
                    logger.info("error");
                    e.printStackTrace();
                }
            }
        }

        public void map(final Object key, final Text value, final Context context) throws IOException, InterruptedException {
            String[] s = value.toString().split("\t");
            double sum = weights.get(0);
            List<Double> weight_op;
            double y = Double.parseDouble(s[0]);
            weight_op = new ArrayList<>();
            weight_op.add(1.0);
            for (int i = 1; i < s.length; i++) {
                sum += weights.get(i) * Double.parseDouble(s[i]);
            }
            double error =  sum-y;
            for (int i = 0; i < weight_up_sums.size(); i++) {
                if (i > 0) {
                    weight_up_sums.put(i, weight_up_sums.get(i) + Double.parseDouble(s[i]) * error);
                } else {
                    weight_up_sums.put(i, weight_up_sums.get(i) + error);
                }
            }
            double cost = Math.pow(error, 2);
            sum_error+=cost;
            nob++;
        }
        public void cleanup(Context context) throws IOException,InterruptedException {
            for (int i = 0; i < weight_up_sums.size(); i++) {
                context.write(new IntWritable(i), new Text(weights.get(i) + "\t" + weight_up_sums.get(i).toString()));
            }
            context.write(new IntWritable(-1),new Text(nob+"\t"+sum_error));
        }
    }

    public static class Train_reducer extends Reducer<IntWritable, Text, Text,Text> {
        private Map<Integer,Double> w_pgd=new HashMap<>();
        private Map<Integer,Double> weights=new HashMap<>();
        double sse=0;
        int nob=0;
        String w;
        public void reduce(final IntWritable key, final Iterable<Text> values, final Context context) throws IOException, InterruptedException {
            for(Text d: values) {
                    String[] s=d.toString().split("\t");
                    if(!key.toString().equals("-1")) {
                        if (!w_pgd.containsKey(Integer.parseInt(key.toString()))) {
                            w_pgd.put(Integer.parseInt(key.toString()), Double.parseDouble(s[1]));
                            weights.put(Integer.parseInt(key.toString()), Double.parseDouble(s[0]));
                        } else {
                            w_pgd.put(Integer.parseInt(key.toString()), Double.parseDouble(s[1]) + w_pgd.get(Integer.parseInt(key.toString())));
                        }
                    }
                    else{
                        sse+=Double.parseDouble(s[1]);
                        nob+=Integer.parseInt(s[0]);
                    }
            }
        }
        public void cleanup(Context context) throws IOException,InterruptedException{
            sse=sse/nob;
            logger.info(sse);
            w="";
            double error=Double.parseDouble(context.getConfiguration().get("prev_error"));
            double lr=Double.parseDouble(context.getConfiguration().get("lr"));
            double tolerance=Double.parseDouble(context.getConfiguration().get("tolerance"));
            if (Math.abs(sse - error) > tolerance && sse<error) {
                for (int i=0;i<weights.size();i++) {
                    weights.put(i, weights.get(i) - 2.0*lr * w_pgd.get(i)/nob);
                    if(i<weights.size()-1)
                        w+=weights.get(i)+",";
                    else
                        w+=weights.get(i);
                }
                context.getCounter(Stats.curr_mse).increment((long)(sse*100000000));
                context.write(new Text("weights"),new Text(w));
            } else {
                context.getCounter(Stats.flag).increment(1);
                flag=1;
            }
        }
    }

    @Override
    public int run(final String[] args) throws Exception {
        while(flag==0) {
            logger.info("Epoch:"+num_it);
            String loc,loc1,del;
            if(num_it==0)
                error=Double.POSITIVE_INFINITY;
            final Configuration conf = getConf();
            conf.setInt("noi",num_it);
            conf.setDouble("lr",0.01);
            conf.setDouble("tolerance",0.00001);
            conf.setDouble("prev_error",error);
            String s3_loc="s3n://mr-manav-bucket-1/";
            num_it++;
            del="";
            final Job job = Job.getInstance(conf, "Linear Regression_train");
            job.setJarByClass(LinearReg.class);
            final Configuration jobConf = job.getConfiguration();
            jobConf.set("mapreduce.output.textoutputformat.separator", ",");
            if(num_it==1)
            {
                loc="output0.txt";
            }
            else{
                loc=args[1]+(num_it-1)+"/part-r-00000";
                del=args[1]+(num_it-1);
            }
            loc1=args[1]+num_it;
            try {
                if(num_it==1)
                job.addCacheFile(new URI(s3_loc+loc+"#"+loc));
                else
                    job.addCacheFile(new URI(loc+"#output"+(num_it-1)+".txt"));
            } catch (Exception e) {
                e.printStackTrace();
            }
            job.setMapperClass(Train_mapper.class);
            job.setReducerClass(Train_reducer.class);
            job.setNumReduceTasks(1);
            job.setMapOutputKeyClass(IntWritable.class);
            job.setMapOutputValueClass(Text.class);
            job.setOutputKeyClass(Text.class);
            job.setOutputValueClass(Text.class);
            FileInputFormat.addInputPath(job, new Path(args[0]));
            FileOutputFormat.setOutputPath(job, new Path(loc1));
            if (!job.waitForCompletion(true))
                System.exit(1);
            Counters c=job.getCounters();
            sum_error=c.findCounter(Stats.curr_mse).getValue()/(double)100000000;
            flag=c.findCounter(Stats.flag).getValue();
            if(flag==0)
            {
                error=sum_error;
                if(num_it==1)
                    FileSystem.get(new URI(s3_loc),conf).delete(new Path(s3_loc+loc),true);
                else
                    FileSystem.get(new URI(s3_loc),conf).delete(new Path(del),true);
            }
            else{
                FileSystem.get(new URI(s3_loc),conf).delete(new Path(loc1),true);
            }

            logger.info("the error:"+error);

        }
            return 0;
    }

    public static void main(final String[] args) {
        if (args.length != 2) {
            throw new Error("Three arguments required:\n<input-dir> <output1-dir> <output2-dir>");
        }

        try {
            num_it=0;
            flag=0;
                ToolRunner.run(new LinearReg(), args);
        } catch (final Exception e) {
            logger.error("", e);
        }
    }
}
