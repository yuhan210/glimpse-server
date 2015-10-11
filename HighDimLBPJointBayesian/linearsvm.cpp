#include <stdio.h>
#include <stdlib.h>
#include "linearsvm.h"
#include <string.h>
#include <locale.h>
#include <math.h>
#include <stdarg.h>
#include <iostream>

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

static const char *solver_type_table[]=
{
	"L2R_LR", "L2R_L2LOSS_SVC_DUAL", "L2R_L2LOSS_SVC", "L2R_L1LOSS_SVC_DUAL", "MCSVM_CS",
	"L1R_L2LOSS_SVC", "L1R_LR", "L2R_LR_DUAL",
	"", "", "",
	"L2R_L2LOSS_SVR", "L2R_L2LOSS_SVR_DUAL", "L2R_L1LOSS_SVR_DUAL", NULL
};

struct model *load_model(const char *model_file_name)
{
	FILE *fp = fopen(model_file_name,"r");
	if(fp==NULL) return NULL;

	int i;
	int nr_feature;
	int n;
	int nr_class;
	double bias;
	model *model_ = Malloc(model,1);
	parameter& param = model_->param;

	model_->label = NULL;

	char *old_locale = strdup(setlocale(LC_ALL, NULL));
	setlocale(LC_ALL, "C");

	char cmd[81];
	while(1)
	{
		fscanf(fp,"%80s",cmd);
		if(strcmp(cmd,"solver_type")==0)
		{
			fscanf(fp,"%80s",cmd);
			int i;
			for(i=0;solver_type_table[i];i++)
			{
				if(strcmp(solver_type_table[i],cmd)==0)
				{
					param.solver_type=i;
					break;
				}
			}
			if(solver_type_table[i] == NULL)
			{
				fprintf(stderr,"unknown solver type.\n");

				setlocale(LC_ALL, old_locale);
				free(model_->label);
				free(model_);
				free(old_locale);
				return NULL;
			}
		}
		else if(strcmp(cmd,"nr_class")==0)
		{
			fscanf(fp,"%d",&nr_class);
			model_->nr_class=nr_class;
		}
		else if(strcmp(cmd,"nr_feature")==0)
		{
			fscanf(fp,"%d",&nr_feature);
			model_->nr_feature=nr_feature;
		}
		else if(strcmp(cmd,"bias")==0)
		{
			fscanf(fp,"%lf",&bias);
			model_->bias=bias;
		}
		else if(strcmp(cmd,"w")==0)
		{
			break;
		}
		else if(strcmp(cmd,"label")==0)
		{
			int nr_class = model_->nr_class;
			model_->label = Malloc(int,nr_class);
			for(int i=0;i<nr_class;i++)
				fscanf(fp,"%d",&model_->label[i]);
		}
		else
		{
			fprintf(stderr,"unknown text in model file: [%s]\n",cmd);
			setlocale(LC_ALL, old_locale);
			free(model_->label);
			free(model_);
			free(old_locale);
			return NULL;
		}
	}

	nr_feature=model_->nr_feature;
	if(model_->bias>=0)
		n=nr_feature+1;
	else
		n=nr_feature;
	int w_size = n;
	int nr_w;
	if(nr_class==2 && param.solver_type != MCSVM_CS)
		nr_w = 1;
	else
		nr_w = nr_class;

	model_->w=Malloc(double, w_size*nr_w);
	for(i=0; i<w_size; i++)
	{
		int j;
		for(j=0; j<nr_w; j++)
			fscanf(fp, "%lf ", &model_->w[i*nr_w+j]);
		fscanf(fp, "\n");
	}

	setlocale(LC_ALL, old_locale);
	free(old_locale);

	if (ferror(fp) != 0 || fclose(fp) != 0) return NULL;

	return model_;
}

int check_probability_model(const struct model *model_)
{
	return (model_->param.solver_type==L2R_LR ||
			model_->param.solver_type==L2R_LR_DUAL ||
			model_->param.solver_type==L1R_LR);
}

double get_prediction(const struct model *model_, const struct feature_node *x, std::vector<double> confidence_est){
	int nr_class=model_->nr_class;
	int dec_max_idx = 0;
	int i;
	for(i=1;i<nr_class;i++)
	{
		if(confidence_est.at(i) > confidence_est.at(dec_max_idx))
			dec_max_idx = i;
	}
	return model_->label[dec_max_idx];
}

double predict_values(const struct model *model_, const struct feature_node *x, std::vector<double>& confidence_est, int& dec_max_idx)
{
	int idx;
	int n;
	if(model_->bias>=0)
		n=model_->nr_feature+1;
	else
		n=model_->nr_feature;
	double *w=model_->w;
	int nr_class=model_->nr_class;
	int i;
	int nr_w;
	if(nr_class==2 && model_->param.solver_type != MCSVM_CS)
		nr_w = 1;
	else
		nr_w = nr_class;

	const feature_node *lx=x;
	/**for(i=0;i<nr_w;i++){		
		confidence_est.at(i) = 0;
	}**/
	for(; (idx=lx->index)!=-1; lx++) // for each feature dim
	{

		// the dimension of testing data may exceed that of training
		if(idx<=n)
			for(i=0;i<nr_w;i++){
				confidence_est.at(i) += w[(idx-1)*nr_w+i]*lx->value;
			}
	}

	dec_max_idx = 0;
	for(i=1;i<nr_class;i++)
	{
		if(confidence_est.at(i) > confidence_est.at(dec_max_idx)){
				dec_max_idx = i;
		}			
	}
	return model_->label[dec_max_idx];
	
}

double predict_values(const struct model *model_, const struct feature_node *x, double *dec_values, int& dec_max_idx)
{
	int idx;
	int n;
	if(model_->bias>=0)
		n=model_->nr_feature+1;
	else
		n=model_->nr_feature;
	double *w=model_->w;
	int nr_class=model_->nr_class;
	int i;
	int nr_w;
	if(nr_class==2 && model_->param.solver_type != MCSVM_CS)
		nr_w = 1;
	else
		nr_w = nr_class;

	const feature_node *lx=x;
	for(i=0;i<nr_w;i++)
		dec_values[i] = 0;
	for(; (idx=lx->index)!=-1; lx++) // for each feature dim
	{

		// the dimension of testing data may exceed that of training
		if(idx<=n)
			for(i=0;i<nr_w;i++)
				dec_values[i] += w[(idx-1)*nr_w+i]*lx->value;
	}

	if(nr_class==2)
	{
		if(model_->param.solver_type == L2R_L2LOSS_SVR ||
		   model_->param.solver_type == L2R_L1LOSS_SVR_DUAL ||
		   model_->param.solver_type == L2R_L2LOSS_SVR_DUAL)
			return dec_values[0];
		else
			return (dec_values[0]>0)?model_->label[0]:model_->label[1];
	}
	else
	{
		
		for(i=1;i<nr_class;i++)
		{
			if(dec_values[i] > dec_values[dec_max_idx])
				dec_max_idx = i;
		}
		return model_->label[dec_max_idx];
	}
}


double predict_probability(const struct model *model_, const struct feature_node *x, double* prob_estimates, double& prob)
{
	if(check_probability_model(model_))
	{
		int i;
		int nr_class=model_->nr_class;
		int nr_w;
		if(nr_class==2)
			nr_w = 1;
		else
			nr_w = nr_class;

		int max_index = 0;
		double label=predict_values(model_, x, prob_estimates, max_index);
		for(i=0;i<nr_w;i++)
			prob_estimates[i]=1/(1+exp(-prob_estimates[i]));

		if(nr_class==2) // for binary classification
			prob_estimates[1]=1.-prob_estimates[0];
		else
		{
			double sum=0;
			for(i=0; i<nr_class; i++)
				sum+=prob_estimates[i];

			for(i=0; i<nr_class; i++){
				prob_estimates[i]=prob_estimates[i]/sum;
			}
			prob = prob_estimates[max_index];
		}

		return label;
	}
	else
		return 0;
}

double predict_probability(const struct model *model_, const struct feature_node *x, std::vector<double>& confidence_est, double& prob)
{
	if(check_probability_model(model_))
	{
		int i;
		int nr_class=model_->nr_class;
		int nr_w;
		if(nr_class==2)
			nr_w = 1;
		else
			nr_w = nr_class;

		int max_index = 0;
		double label=predict_values(model_, x, confidence_est, max_index);
		
		double sum=0;
		for(i=0;i<nr_w;i++){
			confidence_est.at(i) = 1/(1+exp(-confidence_est.at(i)));
			sum += confidence_est.at(i);
		}
		
		/**double sum=0;
		for(i=0; i<nr_class; i++){
			sum += confidence_est.at(i);
		}**/

		for(i=0; i<nr_class; i++){
			confidence_est.at(i) = confidence_est.at(i)/sum;
		}
		prob = confidence_est.at(max_index);
		return label;
	}
	else
		return 0;
}

int get_nr_feature(const model *model_)
{
	return model_->nr_feature;
}

int get_nr_class(const model *model_)
{
	return model_->nr_class;
}

void get_labels(const model *model_, int* label)
{
	if (model_->label != NULL)
		for(int i=0;i<model_->nr_class;i++)
			label[i] = model_->label[i];
}

void free_and_destroy_model(struct model **model_ptr_ptr)
{
	struct model *model_ptr = *model_ptr_ptr;
	if(model_ptr != NULL)
	{
		free_model_content(model_ptr);
		free(model_ptr);
	}
}
void free_model_content(struct model *model_ptr)
{
        if(model_ptr->w != NULL)
                free(model_ptr->w);
        if(model_ptr->label != NULL)
                free(model_ptr->label);
}
         
