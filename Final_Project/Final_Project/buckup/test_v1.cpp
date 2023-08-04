#include <LEDA/graph/graph.h>
#include <LEDA/graph/mw_matching.h>
#include <LEDA/graph/ugraph.h>
#include <glpk.h>
#include <bitset>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace leda;
class kiss {
   public:
    int _num_input;
    int _num_output;
    int _num_transition;
    int _num_state;
    std::string _ini_state;
    void read_kiss_info(int i, int o, int t, int s, std::string is) {
        _num_input = i;
        _num_output = o;
        _num_transition = t;
        _num_state = s;
        _ini_state = is;
    }
    kiss(){};
};

struct node_info {
    node n_in;
    double probability;
};

void kiss_info_parse(std::ifstream &file, kiss &kiss_info) {
    std::stringstream ss;
    std::string line, ignore;
    std::string is;
    int i, o, t, s;
    std::getline(file, line);
    std::getline(file, line);
    ss.clear();
    ss << line;
    ss >> ignore >> i;
    std::getline(file, line);
    ss.clear();
    ss << line;
    ss >> ignore >> o;
    std::getline(file, line);
    ss.clear();
    ss << line;
    ss >> ignore >> t;
    std::getline(file, line);
    ss.clear();
    ss << line;
    ss >> ignore >> s;
    std::getline(file, line);
    ss.clear();
    ss << line;
    ss >> ignore >> is;
    // std::cerr << "i: " << i << '\n';
    // std::cerr << "o: " << o << '\n';
    // std::cerr << "t: " << t << '\n';
    // std::cerr << "s: " << s << '\n';
    // std::cerr << "is: " << is << '\n';
    kiss_info.read_kiss_info(i, o, t, s, is);
    // std::cerr << kiss_info._num_input << " " << kiss_info._num_output << '\n';
}

void construct_graph(std::ifstream &file, const kiss &kiss_info, GRAPH<std::string, double> &G, std::map<std::string, node_info *> &node_ptr, std::map<std::string, bool> &node_exist) {
    std::map<std::pair<std::string, std::string>, int> edge_exist;

    std::string line, first_node, second_node, input, source_node, sink_node;
    int weight = 0, dash_count = 0;

    std::getline(file, line);
    for (int i = 0; i < kiss_info._num_input; i++) {
        if (line[i] == '-')
            dash_count++;
    }
    weight = pow(2, dash_count);
    std::stringstream ss;
    ss << line;
    ss >> input >> first_node >> second_node;
    auto edgepair = std::make_pair(first_node, second_node);
    edge_exist[edgepair] = weight;
    source_node = first_node;
    sink_node = second_node;

    for (int transition = 1; transition < kiss_info._num_transition; transition++) {
        std::getline(file, line);
        dash_count = 0;
        weight = 0;
        for (int i = 0; i < kiss_info._num_input; i++) {
            if (line[i] == '-')
                dash_count++;
        }
        weight = pow(2, dash_count);
        std::stringstream ss;
        ss << line;
        ss >> input >> first_node >> second_node;
        node_exist.emplace(first_node, true);
        node_exist.emplace(second_node, true);
        auto edgepair = std::make_pair(first_node, second_node);
        auto exist = edge_exist.find(edgepair);

        if (exist != edge_exist.end()) {
            edge_exist[edgepair] += weight;
        } else {
            edge_exist[edgepair] = weight;
        }
    }
    for (const auto &i : node_exist) {
        node n = G.new_node(i.first);
        // std::cout << G[n] << '\n';
        // std::cout << &n << '\n';
        node_info *n_info = new node_info;
        n_info->n_in = n;
        n_info->probability = 0;
        node_ptr[i.first] = n_info;
    }
    // for(const auto &i : node_ptr){
    // 	std::cout << G[i.second] << '\n';
    // }
    // node v;
    // forall_nodes(v, G)
    // {
    // 	std::cout << G[v] << '\n';
    // }

    for (const auto &i : edge_exist) {
        std::string source = i.first.first;
        std::string sink = i.first.second;
        double edge_weight = i.second;
        // std::cout << G[*node_ptr[source]] << " " << G[*node_ptr[sink]] << '\n';
        edge e = G.new_edge(node_ptr[source]->n_in, node_ptr[sink]->n_in, edge_weight);
    }

    // edge e;
    //  forall_edges(e,G){
    //  	G.print_edge(e);
    //  	std::cout << '\n';
    //  }
    //  std::cout << "00000000000000000000000000000000000000" << '\n';
}

void compute_stable_propobility(const kiss &kiss_info, GRAPH<std::string, double> &G, std::map<std::string, node_info *> &node_ptr, std::map<std::string, bool> &node_exist, std::map<std::string, double> &stable_probability) {
    double **transition_matrix = new double *[kiss_info._num_state];
    for (int i = 0; i < kiss_info._num_state; i++) {
        transition_matrix[i] = new double[kiss_info._num_state];
    }
    for (int i = 0; i < kiss_info._num_state; i++) {
        memset(transition_matrix[i], 0, kiss_info._num_state * sizeof(double));
    }
    // for (int i = 0; i < kiss_info._num_state; i++) {
    //     for (int j = 0; j < kiss_info._num_state; j++) {
    //         std::cout << transition_matrix[i][j] << " ";
    //     }
    //     std::cout << std::endl;
    // }
    std::unordered_map<std::string, int> stateMap;
    int map_num = 0;
    for (const auto &i : node_exist) {
        stateMap[i.first] = map_num;
        map_num++;
    }
    // for (const auto& i : stateMap) {
    // 	std::cout << i.first << " " << i.second << '\n';
    // }

    node v;
    forall_nodes(v, G) {
        // std::cout << "G[v]  : " << G[v] << '\n';
        edge e;
        double total = 0;
        forall_out_edges(e, v) {
            node source_node = source(e);
            node target_node = target(e);
            double weight = G[e];
            total += weight;
        }
        forall_out_edges(e, v) {
            node source_node = source(e);
            node target_node = target(e);
            double weight = G[e];
            transition_matrix[stateMap[G[target_node]]][stateMap[G[source_node]]] = weight / total;
            G.assign(e, weight / total);
            // std::cout << G[source_node] << " " << G[e] << " " << G[target_node] << '\n';
        }
        // std::cout << "///////////////////////////" << '\n';
    }

    // for (int i = 0; i < kiss_info._num_state; i++) {
    //     for (int j = 0; j < kiss_info._num_state; j++) {
    //         std::cout << transition_matrix[i][j] << " ";
    //     }
    //     std::cout << std::endl;
    // }

    glp_prob *lp;
    lp = glp_create_prob();
    glp_set_prob_name(lp, "compute_stable_propobility");

    // define variable
    glp_add_cols(lp, kiss_info._num_state);
    for (int i = 1; i <= kiss_info._num_state; i++) {
        std::string variable_name = "state" + std::to_string(i - 1);
        glp_set_col_name(lp, i, variable_name.c_str());
        glp_set_col_kind(lp, i, GLP_CV);  // continous variable
        glp_set_col_bnds(lp, i, GLP_DB, 0.0, 1.0);
    }
    // define constraint bound
    glp_add_rows(lp, kiss_info._num_state + 1);
    for (int i = 1; i <= kiss_info._num_state; i++) {
        std::string variable_name = "new_state" + std::to_string(i - 1);
        glp_set_row_name(lp, i, variable_name.c_str());
        glp_set_row_bnds(lp, i, GLP_FX, 0.0, 0.0);
    }
    glp_set_row_name(lp, kiss_info._num_state + 1, "sum");
    glp_set_row_bnds(lp, kiss_info._num_state + 1, GLP_FX, 1.0, 1.0);

    glp_set_obj_name(lp, "value");
    glp_set_obj_dir(lp, GLP_MIN);
    for (int i = 1; i <= kiss_info._num_state; i++) {
        glp_set_obj_coef(lp, i, 1.0);
    }

    int *ind = new int[kiss_info._num_state + 1];
    double *val = new double[kiss_info._num_state + 1];

    for (int i = 0; i < kiss_info._num_state; i++) {
        memset(ind, 0, sizeof(int) * (kiss_info._num_state + 1));
        memset(val, 0, sizeof(double) * (kiss_info._num_state + 1));

        // std::cout << "total : " << total << '\n';
        for (int j = 0; j < kiss_info._num_state; j++) {
            double total = 0;
            for (int k = 0; k < kiss_info._num_state; k++) {
                total += transition_matrix[k][j];
            }
            ind[j + 1] = j + 1;
            if (transition_matrix[i][j] != 0) {
                // std::cout << transition_matrix[i][j] << '\n';
                val[j + 1] = transition_matrix[i][j];
                // std::cout << double(transition_matrix[i][j]) << '\n';
            }
            if (i == j)
                val[j + 1] -= 1;
            // std::cout << "ind[j + 1]: " << ind[j + 1] << " val[j + 1]: " << val[j + 1] << '\n';
        }
        // for (int j = 0; j < kiss_info._num_state+1; j++) std::cout << ind[j] << ' ' << val[j] << '\n';
        //  std::cout << "////////////////////////////////////" << '\n';
        glp_set_mat_row(lp, i + 1, kiss_info._num_state, ind, val);
    }
    for (int j = 0; j < kiss_info._num_state; j++) {
        ind[j + 1] = j + 1;
        val[j + 1] = 1;
    }
    // for (int j = 0; j < kiss_info._num_state+1; j++) std::cout << ind[j] << ' ' << val[j] << '\n';
    glp_set_mat_row(lp, kiss_info._num_state + 1, kiss_info._num_state, ind, val);
    glp_simplex(lp, nullptr);

    // for (int i = 0; i < kiss_info._num_state; i++)
    // {
    // 	std::cout << glp_get_col_prim(lp, i + 1) << '\n';
    // }

    int glp_get_col_prim_index = 1;
    for (const auto &i : node_exist) {
        stable_probability[i.first] = glp_get_col_prim(lp, glp_get_col_prim_index);
        glp_get_col_prim_index++;
    }

    // std::ofstream outfile("output.out");
    // // 輸出問題資訊
    // outfile << "Problem:    " << glp_get_prob_name(lp) << "\n";
    // outfile << "Rows:       " << glp_get_num_rows(lp) << "\n";
    // outfile << "Columns:    " << glp_get_num_cols(lp) << "\n";
    // outfile << "Non-zeros:  " << glp_get_num_nz(lp) << "\n";
    // outfile << "Status:     " << (glp_get_status(lp) == GLP_OPT ? "OPTIMAL" : "NOT OPTIMAL") << "\n";
    // outfile << "Objective:  value = " << glp_get_obj_val(lp) << " (" << (glp_get_obj_dir(lp) == GLP_MIN ? "MINimum" : "MAXimum") << ")\n";

    // // 輸出行資訊
    // outfile << "\n";
    // outfile << "   No.   Row name   St   Activity     Lower bound   Upper bound\n";
    // outfile << "------ ------------ -- ------------- ------------- -------------\n";
    // for (int i = 1; i <= glp_get_num_rows(lp); i++) {
    //     outfile << "     " << i << " " << glp_get_row_name(lp, i) << "   ";
    //     outfile << (glp_get_row_stat(lp, i) == GLP_BS ? "B " : "NS");
    //     outfile << "   " << glp_get_row_prim(lp, i) << "   ";
    //     outfile << glp_get_row_lb(lp, i) << "   ";
    //     outfile << glp_get_row_ub(lp, i) << "\n";
    // }

    // // 輸出列資訊
    // outfile << "\n";
    // outfile << "   No. Column name  St   Activity     Lower bound   Upper bound\n";
    // outfile << "------ ------------ -- ------------- ------------- -------------\n";
    // for (int j = 1; j <= glp_get_num_cols(lp); j++) {
    //     outfile << "     " << j << " " << glp_get_col_name(lp, j) << "   ";
    //     outfile << (glp_get_col_stat(lp, j) == GLP_BS ? "B " : "NS");
    //     outfile << "   " << glp_get_col_prim(lp, j) << "   ";
    //     outfile << glp_get_col_lb(lp, j) << "   ";
    //     outfile << glp_get_col_ub(lp, j) << "\n";
    // }

    // // 關閉檔案
    // outfile.close();
}

void construct_undirect_weight_graph(const kiss &kiss_info, GRAPH<std::string, double> &G, const std::map<std::string, node_info *> &node_ptr, std::map<std::string, double> &stable_probability) {
    node v;
    forall_nodes(v, G) {
        edge e;
        forall_out_edges(e, v) {
            double transition_probabilities = G[e] * stable_probability[G[v]];
            G.assign(e, transition_probabilities);
        }
    }

    // forall_nodes(v, G)
    // {
    // 	std::cout << "v: " << G[v] << '\n';
    // 	edge e;
    // 	forall_out_edges(e, v)
    // 	{
    // 		std::cout << G[e] << "\n";
    // 	}
    // }

    G.make_undirected();
    G.make_directed();

    forall_nodes(v, G) {
        edge e;
        forall_out_edges(e, v) {
            node target_node = target(e);
            edge target_edge;
            forall_out_edges(target_edge, target_node) {
                if (target(target_edge) == v) {
                    G.assign(e, G[e] + G[target_edge]);
                    G.del_edge(target_edge);
                }
            }
        }
    }
    G.make_undirected();
    // forall_nodes(v, G){
    // 	edge e;
    // 	std::cout << "v: " << G[v] << '\n';
    // 	forall_out_edges(e, v){
    // 		std::cout << G[e] << "\n";
    // 	}
    // }
    // edge e;
    // forall_edges(e, G)
    // {
    // 	std::cout << G[e] << '\n';
    // }
}

void Max_matching(list<edge> &Matching_edge_result, GRAPH<std::string, double> &G) {
    edge_array<int> weight(G);
    Matching_edge_result = MAX_WEIGHT_MATCHING(G, weight);
    // std::cout << "Maximum Weighted Matching:" << std::endl;
    double weight_M = 0;
    edge e;
    // forall_edges(e, G)
    // {
    // 	std::cout << G[source(e)] << "----" << G[e] << "----" << G[target(e)] << '\n';
    // }
    // std::cout << "////////////////////////////" << '\n';
    forall(e, Matching_edge_result) {
        // std::cout << G[source(e)] << "----" << G[e] << "----" << G[target(e)] << '\n';
        weight_M += weight[e];
    }
    // std::cout << "////////////////////////////" << '\n';
    // std::cout << " weight: " << weight_M << std::endl;
}

void Encode_state(const kiss &kiss_info, GRAPH<std::string, double> &G, list<edge> &Matching_edge_result, std::map<std::string, std::string> &state_encode_bit_string, std::map<std::string, bool> &node_exist) {
    double state_num = kiss_info._num_state;
    int bit_string_length = std::ceil(std::log2(state_num));
    // std::cout << "bit_string_length: " << bit_string_length << '\n';
    int encode_int = 0;
    for (auto &i : Matching_edge_result) {
        node source_node = source(i);
        // std::cout << "encode_int: " << encode_int << '\n';
        std::bitset<sizeof(int) * 8> binary1(encode_int);
        std::string binaryString1 = binary1.to_string();
        state_encode_bit_string[G[source_node]] = binaryString1.substr(sizeof(int) * 8 - bit_string_length);
        node_exist[G[source_node]] = false;
        encode_int++;
        // std::cout << "encode_int: " << encode_int << '\n';
        node target_node = target(i);
        std::bitset<sizeof(int) * 8> binary2(encode_int);
        std::string binaryString2 = binary2.to_string();
        state_encode_bit_string[G[target_node]] = binaryString2.substr(sizeof(int) * 8 - bit_string_length);
        node_exist[G[target_node]] = false;
        encode_int++;
    }
    for (auto &i : node_exist) {
        if (i.second) {
            // std::cout<< "////////////////////////////////////// " << '\n';
            // std::cout << "encode_int: " << encode_int << '\n';
            std::bitset<sizeof(int) * 8> binary(encode_int);
            std::string binaryString = binary.to_string();
            state_encode_bit_string[i.first] = binaryString.substr(sizeof(int) * 8 - bit_string_length);
            encode_int++;
        }
    }
}

int calculateCost(const std::string &str1, const std::string &str2) {
    int cost = 0;
    for (std::size_t i = 0; i < str1.length(); ++i) {
        if (str1[i] != str2[i]) {
            cost++;
        }
    }
    return cost;
}

std::string executeCommand(const std::string &command, const std::string &filename) {
    std::string tempFileName = filename + ".txt";
    std::string redirectCommand = command + " > " + tempFileName;
    int result = std::system(redirectCommand.c_str());
    std::ifstream tempFile(tempFileName);
    std::string output = "";
    if (tempFile.is_open()) {
        std::string line;
        while (std::getline(tempFile, line)) {
            output += line + "\n";
        }
        tempFile.close();
        std::remove(tempFileName.c_str());
    } else {
        std::cerr << "無法讀取臨時文件\n";
    }
    return output;
}

double sis(const std::string &filename) {
    std::string tclScript =
        "read_blif ../output/" + filename + ".blif\n"
        "source /users/student/mr111/lywu22/ALS/Final_Project/sis_script/opt_map_power.scr\n";
    std::string tcl_name = "../TCL/" + filename + ".tcl";
    std::ofstream scriptFile(tcl_name);
    if (scriptFile.is_open()) {
        scriptFile << tclScript;
        scriptFile.close();
        // std::cout << "Tcl腳本已成功寫入檔案 filename.tcl\n";
    } else {
        std::cerr << "無法寫入檔案 filename.tcl\n";
    }
    std::string command = "/users/student/mr111/lywu22/ALS/Final_Project/sis-1.3.6-bin/bin/sis -xf " + tcl_name;
    std::string output = executeCommand(command, filename);
    // std::cout << "output" << output << '\n';
    std::string powerPrefix = "Power = ";
    double cost;
    size_t powerPos = output.find(powerPrefix);
    if (powerPos != std::string::npos) {
        powerPos += powerPrefix.length();
        std::string powerString;
        while (powerPos < output.length() && isdigit(output[powerPos]) || output[powerPos] == '.') {
            powerString += output[powerPos];
            powerPos++;
        }
        std::cout << "powerString: " << powerString << '\n';
        cost = std::stod(powerString);
    }
    return cost;
}

std::string getFilenameWithoutExtension(const std::string &filename) {
    size_t lastSlash = filename.find_last_of('/');
    std::string Filename = filename.substr(lastSlash + 1);
    size_t firstDot = Filename.find_first_of('.');
    std::string desiredName = Filename.substr(0, firstDot);
    return desiredName;
}

void write_MYFSM(const std::string &inputFilename, const std::string &Filename, const std::map<std::string, std::string> &Encoding) {
    std::ifstream inputFile(inputFilename);
    std::string outputFilename = "../output/" + Filename + ".blif";
    // std::cout << "outputFilename: " << outputFilename << '\n';
    std::ofstream outputFile(outputFilename);  // 打开输出文件流
    if (!outputFile) {
        std::cerr << "can't construct file : " << outputFilename << std::endl;
        return;
    }
    outputFile << ".mymodel myFSM\n";
    char c;
    while (inputFile.get(c)) {
        outputFile.put(c);
    }
    inputFile.close();
    for (auto &i : Encoding) {
        outputFile << ".code " << i.first << " " << i.second << '\n';
    }
    outputFile << ".end";
    outputFile.close();
}

double Costfromsis(const std::string &filename) {
    double cost = sis(filename);
    return cost;
}

// Generate a random swap move
void generateRandomMove(int &i, int &j, int size, std::mt19937 &rng) {
    std::uniform_int_distribution<int> dist(0, size - 1);
    i = dist(rng);
    do {
        j = dist(rng);
    } while (i == j);
}

// Apply simulated annealing to optimize FSM encoding
std::map<std::string, std::string> simulatedAnnealing(const std::map<std::string, std::string> &initialEncoding, 
    double initialTemperature, double coolingRate, std::mt19937 &rng, const std::string &inputFilename, const std::string &Filename, std::ofstream &costFile) {

    std::map<std::string, std::string> currentEncoding = initialEncoding;
    std::map<std::string, std::string> bestEncoding = initialEncoding;
    write_MYFSM(inputFilename, Filename, initialEncoding);
    double currentCost = Costfromsis(Filename);
    double bestCost = currentCost;
    double temperature = initialTemperature;
    double reject = 0;
    double swap_solution_count = 0;
    // std::cout << "initial_cost : " << currentCost << '\n';
    std::cout << "Start SA" << '\n';
    for(int a = 0; a < 1000; a++){
    // while (temperature > 0.1 && (reject / swap_solution_count) <= 0.95) {
        // std::cout << "temperature : " << temperature << '\n';
        // std::cout << "reject rate : " << reject / swap_solution_count << '\n';
        // swap_solution_count = 0;
        int uphill = 0;
        // for(int o = 0; o < 100; o++){
        // while (uphill < currentEncoding.size() * 20 || swap_solution_count < currentEncoding.size() * 40) {
            int i, j;
            generateRandomMove(i, j, currentEncoding.size(), rng);
            swap_solution_count++;
            // Swap the bit strings of two states
            auto it1 = std::next(currentEncoding.begin(), i);
            auto it2 = std::next(currentEncoding.begin(), j);
            std::swap(it1->second, it2->second);
            write_MYFSM(inputFilename, Filename, currentEncoding);
            double newCost = Costfromsis(Filename);
            costFile << "currentCost : " << currentCost << '\n';
            costFile << "new_Cost : " << newCost << '\n';
            costFile << "best_Cost : " << bestCost << '\n';
            costFile << "reject rate : " << reject / swap_solution_count << '\n';
            costFile << "temperature : " << temperature << '\n';
            costFile << "///////////////////////////////////////////////////////////////" << '\n';
            // Accept the new state according to the Metropolis criterion
            if ( newCost > 100 && ((newCost < currentCost )|| std::exp((currentCost - newCost) / temperature) > std::generate_canonical<double, 10>(rng)) ) {
                currentCost = newCost;
                // Update the best encoding
                if (newCost > currentCost) {
                    ++uphill;
                }
                if (newCost < bestCost) {
                    bestCost = newCost;
                    bestEncoding = currentEncoding;
                }
            } else {
                // Undo the move
                reject++;
                std::swap(it1->second, it2->second);
                write_MYFSM(inputFilename, Filename, currentEncoding);
            }

        // }
        temperature *= coolingRate;
    }
    // Return the best encoding
    return bestEncoding;
}

int main(int argc, char **argv) {
    std::ifstream file(argv[1]);
    std::string Filename = getFilenameWithoutExtension(argv[1]);
    std::ofstream costFile("cost.txt");
    if (!file.is_open()) {
        std::cerr << "open file error!!";
        return 1;
    }
    kiss kiss_info;
    kiss_info_parse(file, kiss_info);
    // std::cerr << kiss_info._num_input << " " << kiss_info._num_output << " " << kiss_info._num_transition << " " << kiss_info._num_state << " " << kiss_info._ini_state << '\n';
    GRAPH<std::string, double> G;
    std::map<std::string, node_info *> node_ptr;
    std::map<std::string, bool> node_exist;
    std::map<std::string, double> stable_probability;
    std::map<std::string, std::string> state_encode_bit_string;
    list<edge> Matching_edge_result;
    construct_graph(file, kiss_info, G, node_ptr, node_exist);
    compute_stable_propobility(kiss_info, G, node_ptr, node_exist, stable_probability);
    construct_undirect_weight_graph(kiss_info, G, node_ptr, stable_probability);
    Max_matching(Matching_edge_result, G);
    Encode_state(kiss_info, G, Matching_edge_result, state_encode_bit_string, node_exist);
    // for (const auto &state : state_encode_bit_string) {
    //     std::cout << state.first << ": " << state.second << std::endl;
    // }
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Simulated Annealing
    double initialTemperature = 100;  // Initial temperature
    double coolingRate = 0.95;      // Cooling rate
    std::random_device rd;
    std::mt19937 rng(rd());
    std::map<std::string, std::string> optimizedEncoding = simulatedAnnealing(state_encode_bit_string, initialTemperature, coolingRate, rng, argv[1], Filename, costFile);
    write_MYFSM(argv[1], Filename, optimizedEncoding);
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    costFile.close();
    return 0;
}


//cd ALS/Final_Project/src/
