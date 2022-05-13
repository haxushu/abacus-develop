#!/bin/bash

##############################
#RUNNING SET
abacus=abacus           #ABACUS running path
thread=1                    #Thread number 
ca=8                        #accuracy for comparing energy, force, and stress
#ncpu=8                      #parallel core number in mpirun, or use the below value 
ncpu=`cat /proc/cpuinfo | grep "cpu cores" | uniq| awk '{print $NF}'`   #the CPU cores in current mathine
ForceRun=0                  #if ForceRun = 0, before run a job, it will check the file result.log, and if this file 
                            #is normal end, this job will be skipped

DoAllExampleRun=1          #if run allexample in allcase file or not, 0: not do, others: do

DoNCpuTest=0              #if do CPU parallel test or not. 0: not do, others: do
CpuTestExample="P000_si16_pw " #the example used to do NCpu test
NCpuList="2 4 8 16 32"   #the ncore list used to do NCpu test, not run when it is larger than the maxcpucore
maxcpucore=$ncpu

DoKParTest=0               #if do kpar test for PW method
KParTestExample="P010_si2_pw" #do not modify
KParList="2 4 8 16 32"         #kpar setting list, not run when it is larger than maxkpar
maxkpar=$ncpu              #note: maxkpar should less than ncpu and also less than the total K-POINTS number

DoBxyzTest=0               #if modify bx=by=bz value, the default value is 2
BxyzTestExample="P100_si16_lcao" #in this case the grid is 96 x 96 x96, only lcao case
BxyzList="2 3 4 6 8" #the grid number shold be divisible by these values, and these value should also less than the sqrt of grid  
##############################

##############################
#def function check_out
#check_out result.dat
check_out(){
    outfile=$1
    properties=`awk '{print $1}' $outfile`
    for key in $properties; do
        cal=`grep "$key" $outfile | awk '{printf "%.'$ca'f\n",$2}'`
        ref=`grep "$key" result.ref | awk '{printf "%.'$ca'f\n",$2}'`
        deviation=`awk 'BEGIN {x='$ref';y='$cal';if (x<y) {a=y-x} else {a=x-y};printf "%.'$ca'f\n",a}'`
        deviation1=`awk 'BEGIN{print '$deviation'*(10**'$ca')}'`

        if [ $key == "totaltimeref" ]; then
            break
        fi

        if [ ! -n "$deviation" ]; then
            echo "    Error: Fatal Error!"
            let failed++
            currentfolder=`pwd| awk -F '/' '{print $NF}'`
            failedfile="${failedfile}${currentfolder}\n" 
            break
        else
            if [ $(echo "$deviation1 < 1"|bc) = 0 ]; then
                echo "    Error: FAILED!"
                echo "    Error:$key cal=$cal ref=$ref deviation=$deviation"
                let failed++
                currentfolder=`pwd| awk -F '/' '{print $NF}'`
                failedfile="${failedfile}${currentfolder}\n"
                break
            fi
        fi
        let ok++
    done
}
##############################

##############################
#run abacus function
#run_abacus $ncpu $threads $abacuspath $workfolder
run_abacus(){
    cd $3
    printf "Running %-20s: " "$3"

    starttime=`date +'%Y-%m-%d %H:%M:%S'`
    #if there has result.log file, assign lastword="the first word of last line of result.log"
    #and if lastword is equal to "SEE", we think this job is normal end, and it will be 
    #skipped when ForceRun is set to be 0
    lastword=""
    if [[ $ForceRun == 0  && -f result.log ]];then
        lastword=`tail -1 result.log | awk '{print $1}'`
    fi
    if [[ $lastword != "SEE" ]];then
        mpirun -n $1 -env OMP_NUM_THREADS=$2 $abacus > result.log
    else
        printf "**result.log is normal end, skip this job** "
    fi
    endtime=`date +'%Y-%m-%d %H:%M:%S'`
    start_seconds=$(date --date="$starttime" +%s)
    end_seconds=$(date --date="$endtime" +%s)
    echo "Use time "$((end_seconds-start_seconds))"s"

    #runing catch_properties.sh script to collect energy, force and stress to file result.out
    #Then use function "check_out" to compare the results in result.out and result.ref
    bash ../catch_properties.sh result.out  
    check_out result.out
    cd ..
}
##############################

##############################
#Begin to RUN
test -e allcase || echo "Plese specify tests"
test -e allcase || exit 0
which $abacus > /dev/null || echo "Error! ABACUS path was wrong!!"
which $abacus > /dev/null || exit 0

test -f sumall.dat && mv sumall.dat sumall.dat.old
cat version >> sumall.dat
date >> sumall.dat
cat /proc/version >> sumall.dat
cat /proc/cpuinfo | grep "model name"| tail -1| cut -d ':' -f 2 >> sumall.dat
echo "ABACUS path: $abacus" >> sumall.dat
echo "Number of Cores: $ncpu" >> sumall.dat
echo "Accuracy: $ca" >> sumall.dat
echo >> sumall.dat
echo "!!!all data will be summarized in file sumall.dat" >> sumall.dat
echo "" >> sumall.dat
cat sumall.dat

failed=0
failedfile=''
ok=0

##############################
#run all examples in allcase
if [[ $DoAllExampleRun != 0 ]];then
    echo "run all examples in allcase"
    for i in `cat allcase`;do
        test -d $i || continue
        run_abacus $ncpu $thread $i
    done

    #sum the critical timing information
    bash sumdat.sh allcase sum.dat
    echo "##AllExampleRun" >> sumall.dat
    cat sum.dat >> sumall.dat
    echo "" >> sumall.dat
    echo !!!all data are summarized in sum.dat
    echo
fi
##############################
#do NCpu test
if [[ $DoNCpuTest != 0 ]];then
    echo "do N CPU test"
    cpucore=$maxcpucore
    test -f allcase.ncpu && rm allcase.ncpu
    for case in $CpuTestExample;do
        if [[ ! -d $case ]];then
            echo "Error: do NCpu test, can not find $case"
            continue
        fi

        for i in $NCpuList;do
            if [[ $i -lt $cpucore ]];then
                echo ${case}_${i}cpu >> allcase.ncpu
                test -d ${case}_${i}cpu || mkdir ${case}_${i}cpu
                cp ${case}/INPUT ${case}/STRU ${case}/KPT ${case}/result.ref ${case}_${i}cpu

                run_abacus $i $thread ${case}_${i}cpu 
            fi
        done

        echo $case >> allcase.ncpu
        if [[ ! -f $case/result.log ]];then
            run_abacus $ncpu $thread ${case}
        fi
    done

    #sum the critical timing information
    bash sumdat.sh allcase.ncpu sum.dat.ncpu
    echo "##NCpuTest" >> sumall.dat
    cat sum.dat.ncpu >> sumall.dat
    echo "" >> sumall.dat
    echo "!!!all data are summarized in sum.dat.ncpu"
    echo 
fi

##############################
#do KPar test
if [[ $DoKParTest != 0 ]];then
    echo "do N KPar test"
    test -f allcase.kpar && rm allcase.kpar
    for case in $KParTestExample;do
        if [[ ! -d $case ]];then
            echo "Error: do KPar test, can not find $case"
            continue
        fi

        echo $case >> allcase.kpar
        if [[ ! -f $case/result.log ]];then
            run_abacus $ncpu $thread ${case}
        fi

        for i in $KParList;do
            if [[ $i -le $maxkpar ]];then
                echo ${case}_${i}kpar >> allcase.kpar
                test -d ${case}_${i}kpar || mkdir ${case}_${i}kpar
                cp ${case}/INPUT ${case}/STRU ${case}/KPT ${case}/result.ref ${case}_${i}kpar
                echo "kpar $i" >> ${case}_${i}kpar/INPUT

                run_abacus $ncpu $thread ${case}_${i}kpar
            fi
        done
    done
    #sum the critical timing information
    bash sumdat.sh allcase.kpar sum.dat.kpar
    echo "##KParTest" >> sumall.dat
    cat sum.dat.kpar >> sumall.dat
    echo "" >> sumall.dat
    echo "!!!all data are summarized in sum.dat.kpar"
    echo
fi

##############################
#do Bxyz test
if [[ $DoBxyzTest != 0 ]];then
    echo "do bx by bz test"
    test -f allcase.bxyz && rm allcase.bxyz
    for case in $BxyzTestExample;do
        if [[ ! -d $case ]];then
            echo "Error: do BxByBz test, can not find $case"
            continue
        fi

        echo $case >> allcase.bxyz
        if [[ ! -f $case/result.log ]];then
            run_abacus $ncpu $thread ${case}
        fi

        for i in $BxyzList;do
            if [[ $i != 2 ]];then
                echo ${case}_${i}bxyz >> allcase.bxyz
                test -d ${case}_${i}bxyz || mkdir ${case}_${i}bxyz
                cp ${case}/INPUT ${case}/STRU ${case}/KPT ${case}/result.ref ${case}_${i}bxyz
                printf " bx $i \n by $i \n bz $i" >> ${case}_${i}bxyz/INPUT

                run_abacus $ncpu $thread ${case}_${i}bxyz
            fi
        done
    done
    #sum the critical timing information
    bash sumdat.sh allcase.bxyz sum.dat.bxyz
    echo "##BxyzTest" >> sumall.dat
    cat sum.dat.bxyz >> sumall.dat
    echo "" >> sumall.dat
    echo "!!!all data are summarized in sum.dat.bxyz"
    echo
fi

##############################
#Final OUTPUT
if [[ $failed -eq 0 ]];then
    echo "All jobs are finished !!!"
else
    echo "Error: there has $failed jobs are failed!"
    echo -e "$failedfile"
fi
