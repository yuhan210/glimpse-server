#ifndef LINEARSVM_H
#define LINEARSVM_H
#include <vector>

enum { L2R_LR, L2R_L2LOSS_SVC_DUAL, L2R_L2LOSS_SVC, L2R_L1LOSS_SVC_DUAL, MCSVM_CS, L1R_L2LOSS_SVC, L1R_LR, L2R_LR_DUAL, L2R_L2LOSS_SVR = 11, L2R_L2LOSS_SVR_DUAL, L2R_L1LOSS_SVR_DUAL }; /* solver_type */

struct feature_node
{
	int index;
	double value;
};


struct parameter
{
        int solver_type;

        /* these are for training only */
        double eps;             /* stopping criteria */
        double C;
        int nr_weight;
        int *weight_label;
        double* weight;
        double p;
};
     

struct model
{
        struct parameter param;
        int nr_class;           /* number of classes */
        int nr_feature;
        double *w;
        int *label;             /* label of each class */
        double bias;
};
double predict_values(const struct model *model_, const struct feature_node *x, double* dec_values, int& dec_max_idx);
double predict_probability(const struct model *model_, const struct feature_node *x, double* prob_estimates, double& prob);    
double predict_values(const struct model *model_, const struct feature_node *x, std::vector<double>& confidence_est, int& max_dec_index);
double predict_probability(const struct model *model_, const struct feature_node *x, std::vector<double>& confidence_est, double& prob);
struct model *load_model(const char *model_file_name);
int get_nr_feature(const model *model_);
int get_nr_class(const model *model_);
int check_probability_model(const struct model *model_);
void get_labels(const model *model_, int* label);
void free_and_destroy_model(struct model **model_ptr_ptr);
void free_model_content(struct model *model_ptr);
double get_prediction(const struct model *model_, const struct feature_node *x, std::vector<double> confidence_est);
#endif