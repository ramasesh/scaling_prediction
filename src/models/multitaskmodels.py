import torch

class SingleHead(torch.nn.Module):
  """
  Module for a network which attaches a single head (i.e., readout layer)
  to a feature_model
  """

  def __init__(self, feature_model, config):
    """
    feature_model -   model which accepts input and turns into features
    config -          dictionary with keys:
      input_shape     shape of the input which feature_model accepts 
                          (put 1 in the place of the batch)
      n_classes       number of classes
    """
    
    super(SingleHead, self).__init__()

    input_shape = config['input_shape']
    out_features = config['n_classes']

    self.feature_model = feature_model
    self.input_shape = input_shape

    self.num_features = self.__calculate_num_features__()

    self.head = torch.nn.Linear(self.num_features,
                                out_features)

  def __calculate_num_features__(self):
    """
    Calculates the number of output features of the self.feature_model
    """

    with torch.no_grad():
      test_input = torch.zeros(*self.input_shape)
      test_output = self.feature_model(test_input)
      num_features = test_output.view(-1).shape[0]

    return num_features

  def forward(self, x):
    """
    Extracts features from feature_model, then passes those fatures
    through the head
    """

    features = self.feature_model(x)
    features = features.view(features.size(0), -1)
    logits = self.head(features)

    return logits

class MultiHead(torch.nn.Module):
  """
  Module for a network which has multiple heads, i.e. readout layers.
  """

  def __init__(self, feature_model, config):
    """
    input_shape   -   shape of the input which feature_model accepts
    feature_model -   model which accepts input and turns into features
    out_features  -   number of output features desired for each head
    num_heads     -   number of selectable heads
    """

    super(MultiHead, self).__init__()

    input_shape = config['input_shape']
    if isinstance(config['n_classes'], int): # TODO: This could be done automatically from task spec.
        out_features = [config['n_classes']]*config['n_tasks']
    elif isinstance(config['n_classes'], list):
        out_features = config['n_classes']
    else:
        raise TypeError('out_features must be an int or a list.')
        
    self.num_heads = config['n_tasks']

    self.feature_model = feature_model
    self.input_shape = input_shape

    self.num_features = self.__calculate_num_features__()

    self.heads = torch.nn.ModuleDict()
    for i in range(self.num_heads):
      current_head = torch.nn.Linear(self.num_features,
                                     out_features[i])
      self.heads[str(i)] = current_head

    self.__set_active_head__(0)


  def __calculate_num_features__(self):
    """
    Calculates the number of output features of the self.feature_model
    """

    with torch.no_grad():
      test_input = torch.zeros(*self.input_shape)
      test_output = self.feature_model(test_input)
      num_features = test_output.view(-1).shape[0]

    return num_features

  def __set_active_head__(self, head):
    head = str(head)
    self.active_head = head
    for head_num in range(self.num_heads):
      for param in self.heads[str(head_num)].parameters():
        if int(head_num) == int(head):
          param.requires_grad = True
        else:
          param.requires_grad = False

  def __get_active_head__(self, head):
    return self.active_head

  def forward(self, x):
    """
    Extracts features from feature_model, then passes those fatures
    through the currently active head
    """

    features = self.feature_model(x)
    features = features.view(features.size(0), -1)
    logits = self.heads[self.active_head](features)

    return logits
